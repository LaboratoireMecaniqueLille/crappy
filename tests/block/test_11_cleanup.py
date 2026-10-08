# coding: utf-8

"""Critical framework cleanup steps without new ExceptionGroup handling."""

import logging
from multiprocessing import Event
from threading import Thread
from unittest.mock import Mock, patch

from crappy import Block
from crappy._global import CrappyFail
from crappy.blocks.meta_block import block as block_module
from crappy.links import Link

from .block_test_base import BlockTestBase, TestBlock


class TestCleanup(BlockTestBase):
  """Checks that framework failures cannot skip independent resource owners."""

  @staticmethod
  def setup_cleanup() -> None:
    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.logger = Mock()

  def setup_cleanup_resources(self) -> tuple[Mock, ...]:
    """Create owned resources and a Block requiring shutdown escalation."""

    self.setup_cleanup()
    process, link = Mock(), Mock(spec=Link)
    process.name, process.sentinel = 'process', 1
    link.name = 'cleanup-link'
    process.inputs, process.outputs = [link], []
    process.is_alive.return_value = True

    def stop_process() -> None:
      process.is_alive.return_value = False

    process.kill.side_effect = stop_process
    manager, queue, worker = Mock(), Mock(), Mock(spec=Thread)
    worker.ident = 1
    worker.is_alive.return_value = False
    Block._run_blocks = (process,)
    Block.shared_mgr, Block.log_queue, Block.log_thread = manager, queue, worker
    return process, link, manager, queue, worker

  def test_stop_status_logging_failure_does_not_skip_resources(self) -> None:
    """Every stop-status branch preserves cleanup despite failed logging."""

    statuses = ('All Blocks stopped', 'All Blocks stopped after termination',
                'All Blocks stopped after killing',
                'Not all Blocks could be stopped even after killing them')
    for status in statuses:
      for error_type in (OSError, KeyboardInterrupt):
        with self.subTest(status=status, error=error_type.__name__):
          process, link, manager, queue, worker = self.setup_cleanup_resources()
          if status == statuses[0]:
            process.is_alive.return_value = False
          elif status == statuses[1]:
            process.terminate.side_effect = process.kill.side_effect
          elif status == statuses[3]:
            process.kill.side_effect = None
          error = error_type('status log')

          def fail_log(*, level, msg) -> None:
            if msg == status:
              raise error

          Block.logger.log.side_effect = fail_log
          Block.no_raise = True
          with patch.object(block_module.connection, 'wait', return_value=[1]):
            Block._cleanup()

          self.assertIn('log Block stop status', error.__notes__[0])
          link.close.assert_called_once_with()
          manager.shutdown.assert_called_once_with()
          worker.join.assert_called_once_with(timeout=1.0)
          queue.cancel_join_thread.assert_called_once_with()
          queue.close.assert_called_once_with()

  def test_cleanup_logs_escalation_and_resource_shutdown(self) -> None:
    """Logs identify forcibly stopped Blocks and each remaining resource."""

    self.setup_cleanup_resources()
    logger = Block.logger
    with patch.object(block_module.connection, 'wait', return_value=[1]):
      Block._cleanup()

    messages = [(call.kwargs['level'], call.kwargs['msg'])
                for call in logger.log.call_args_list]
    expected = [(logging.WARNING, 'Terminating Block process'),
                (logging.WARNING, 'Killing Block process'),
                (logging.DEBUG, "Closing Link 'cleanup-link'"),
                (logging.INFO, 'Stopping the shared Manager'),
                (logging.INFO, 'Stopping the logging Thread'),
                (logging.DEBUG, 'Closing the log Queue')]
    for message in expected:
      self.assertIn(message, messages)
    positions = [messages.index(message) for message in expected]
    self.assertEqual(positions, sorted(positions))

  def test_cleanup_progress_logging_failure_does_not_skip_resources(
      self) -> None:
    """Every progress message is independent of the operation it announces."""

    messages = ('Terminating Block process', 'Killing Block process',
                "Closing Link 'cleanup-link'", 'Stopping the shared Manager',
                'Stopping the logging Thread', 'Closing the log Queue')
    for message in messages:
      for error_type in (OSError, KeyboardInterrupt):
        with self.subTest(message=message, error=error_type.__name__):
          process, link, manager, queue, worker = self.setup_cleanup_resources()
          failure, interrupt = Block.raise_event, Block.kbi_event
          error = error_type('progress log')

          def fail_log(*, level, msg) -> None:
            if msg == message:
              raise error

          Block.logger.log.side_effect = fail_log
          Block.no_raise = True
          with patch.object(block_module.connection, 'wait', return_value=[1]):
            Block._cleanup()

          process.terminate.assert_called_once_with()
          process.kill.assert_called_once_with()
          link.close.assert_called_once_with()
          manager.shutdown.assert_called_once_with()
          worker.join.assert_called_once_with(timeout=1.0)
          queue.cancel_join_thread.assert_called_once_with()
          queue.close.assert_called_once_with()
          self.assertEqual(failure.is_set(), error_type is OSError)
          self.assertEqual(interrupt.is_set(), error_type is KeyboardInterrupt)
          self.assertTrue(error.__notes__[0].startswith('Block cleanup step:'))

  def test_missing_stop_event_logging_failure_does_not_skip_resources(
      self) -> None:
    """A partial startup still releases resources if its error log fails."""

    for error_type in (OSError, KeyboardInterrupt):
      with self.subTest(error=error_type.__name__):
        process, link, manager, queue, worker = self.setup_cleanup_resources()
        Block.stop_event = None
        error = error_type('missing event log')

        def fail_log(*, level, msg) -> None:
          if msg == "The stop Event should be set but doesn't exist!":
            raise error

        Block.logger.log.side_effect = fail_log
        Block.no_raise = True
        with patch.object(block_module.connection, 'wait', return_value=[1]):
          Block._cleanup()

        self.assertIn('log missing stop Event', error.__notes__[0])
        process.kill.assert_called_once_with()
        link.close.assert_called_once_with()
        manager.shutdown.assert_called_once_with()
        worker.join.assert_called_once_with(timeout=1.0)
        queue.close.assert_called_once_with()

  def test_finish_announcement_failure_cannot_skip_finish(self) -> None:
    """A failed finish log cannot skip the hook or inherited Pipe cleanup."""

    for error_type in (OSError, KeyboardInterrupt):
      with self.subTest(error=error_type.__name__):
        block = TestBlock()
        block._stop_event, block._raise_event, block._kbi_event = (
          Event(), Event(), Event())
        error = error_type('finish log')

        def fail_log(level, msg) -> None:
          if msg == 'Calling the finish method':
            raise error

        with (patch.object(block, '_set_block_logger'),
              patch.object(block, 'log', side_effect=fail_log),
              patch.object(block, 'prepare', side_effect=ValueError('prepare')),
              patch.object(block, 'finish') as finish,
              patch.object(block, '_close_config_connections') as close):
          block.run()

        finish.assert_called_once_with()
        self.assertEqual(close.call_count, 2)
        self.assertEqual(block._kbi_event.is_set(),
                         error_type is KeyboardInterrupt)

  def test_process_failures_do_not_skip_peers_manager_thread_or_queue(
      self) -> None:
    """Failed joins and termination still reach every remaining cleanup."""

    self.setup_cleanup()
    first, second = Mock(), Mock()
    for index, process in enumerate((first, second)):
      process.name = f'process-{index}'
      process.sentinel = index
      process.inputs, process.outputs = [], []
      process.is_alive.return_value = True
      process.kill.side_effect = (
          lambda process=process: setattr(process.is_alive, 'return_value',
                                         False))
    first.join.side_effect = OSError('join')
    first.terminate.side_effect = ValueError('terminate')
    second.terminate.side_effect = (
        lambda: setattr(second.is_alive, 'return_value', False))
    Block._run_blocks = (first, second)
    manager, queue, worker = Mock(), Mock(), Mock(spec=Thread)
    worker.ident = 1
    worker.is_alive.return_value = False
    manager.shutdown.side_effect = RuntimeError('manager')
    Block.shared_mgr, Block.log_queue, Block.log_thread = manager, queue, worker
    failure, logger = Block.raise_event, Block.logger

    with patch.object(block_module.connection, 'wait', return_value=[0, 1]):
      with self.assertRaises(CrappyFail):
        Block._cleanup()

    second.terminate.assert_called_once_with()
    first.kill.assert_called_once_with()
    manager.shutdown.assert_called_once_with()
    worker.join.assert_called_once_with(timeout=1.0)
    queue.cancel_join_thread.assert_called_once_with()
    queue.close.assert_called_once_with()
    self.assertTrue(failure.is_set())
    self.assertGreaterEqual(logger.exception.call_count, 3)
    self.assertEqual(Block._run_blocks, ())
    self.assertIsNone(Block.shared_mgr)

  def test_stop_and_queue_failures_still_run_other_cleanup(self) -> None:
    """Failure to signal stop or cancel a feeder cannot skip later steps."""

    self.setup_cleanup()
    manager, queue = Mock(), Mock()
    Block.stop_event = Mock()
    Block.stop_event.set.side_effect = KeyboardInterrupt()
    queue.cancel_join_thread.side_effect = OSError('cancel')
    Block.shared_mgr, Block.log_queue = manager, queue
    failure, interrupt = Block.raise_event, Block.kbi_event
    Block.no_raise = True
    Block._cleanup()
    manager.shutdown.assert_called_once_with()
    queue.close.assert_called_once_with()
    self.assertTrue(failure.is_set())
    self.assertTrue(interrupt.is_set())

  def test_cleanup_closes_each_link_once_and_still_stops_manager(self) -> None:
    """A Link appearing on both Blocks is cleaned up once, independently."""

    self.setup_cleanup()
    source, target = TestBlock(), TestBlock()
    link = Link(source, target)
    self.addCleanup(link.close)
    manager = Mock()
    Block.shared_mgr = manager
    Block._run_blocks = (source, target)
    with patch.object(link, 'close', side_effect=OSError('link')) as close:
      with self.assertRaises(CrappyFail):
        Block._cleanup()
    close.assert_called_once_with()
    manager.shutdown.assert_called_once_with()

  def test_reporting_failure_still_resets_runtime(self) -> None:
    """An unavailable error reporter must not prevent the final reset."""

    self.setup_cleanup()
    manager, queue = Mock(), Mock()
    Block.shared_mgr, Block.log_queue = manager, queue
    manager.shutdown.side_effect = RuntimeError('shutdown')
    error = OSError('reporting')
    Block.logger.exception.side_effect = error
    with self.assertRaises(OSError) as caught:
      Block._cleanup()
    self.assertIs(caught.exception, error)
    queue.close.assert_called_once_with()
    self.assertIsNone(Block.shared_mgr)
    self.assertIsNone(Block.log_queue)
    self.assertEqual(Block._run_blocks, ())

  def test_stop_event_failure_cannot_skip_finish(self) -> None:
    """The per-Block cleanup hook runs even when signaling stop fails."""

    block = TestBlock()
    block._stop_event = Mock()
    block._stop_event.set.side_effect = OSError('stop event')
    block._raise_event, block._kbi_event = Event(), Event()
    with (patch.object(block, '_set_block_logger'),
          patch.object(block, 'prepare', side_effect=ValueError('prepare')),
          patch.object(block, 'finish') as finish):
      block.run()
    finish.assert_called_once_with()
    self.assertTrue(block._raise_event.is_set())

  def test_inherited_pipe_failure_does_not_skip_peers_or_final_retry(
      self) -> None:
    """Failed inherited endpoints remain available until final cleanup."""

    block = TestBlock()
    block._stop_event = Event()
    block._raise_event, block._kbi_event = Event(), Event()
    first, second = Mock(), Mock()
    first.close.side_effect = [OSError('close'), None]
    block._config_connections_to_close = [first, second]
    with (patch.object(block, '_set_block_logger'),
          patch.object(block, 'prepare') as prepare,
          patch.object(block, 'finish') as finish):
      block.run()
    prepare.assert_not_called()
    finish.assert_called_once_with()
    self.assertEqual(first.close.call_count, 2)
    second.close.assert_called_once_with()
    self.assertEqual(block._config_connections_to_close, [])
