# coding: utf-8

"""Independent framework cleanup and grouped public finalization."""

import logging
from multiprocessing import Event
from threading import Thread
from unittest.mock import Mock, patch

from crappy import Block
from crappy.blocks.meta_block import block as block_module
from crappy.links import Link

from tests.support.blocks import BlockTestBase, TestBlock, exception_leaves


class TestCleanup(BlockTestBase):
  """Checks that framework failures cannot skip independent resource owners."""

  @staticmethod
  def setup_cleanup() -> None:
    Block.reset()
    Block.stop_event = Event()
    Block.logger = Mock()
    Block._ensure_failure_manager()
    Block.run_failure_manager._logger = Block.logger

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
    """Logging failures cannot interrupt any shutdown escalation outcome."""

    for outcome in ('stopped', 'terminated', 'killed', 'survives'):
      for error_type in (OSError, KeyboardInterrupt):
        with self.subTest(outcome=outcome, error=error_type.__name__):
          process, link, manager, queue, worker = self.setup_cleanup_resources()
          if outcome == 'stopped':
            process.is_alive.return_value = False
          elif outcome == 'terminated':
            process.terminate.side_effect = process.kill.side_effect
          elif outcome == 'survives':
            process.kill.side_effect = None
          error = error_type('shutdown logging failure')
          Block.no_raise = True
          with (patch.object(Block.logger, 'log', side_effect=error),
                patch.object(block_module.connection, 'wait', return_value=[1])):
            Block._cleanup()
          self.assertIn(error, [actual for _, actual in
                                Block.run_failure_manager.snapshot()])
          link.close.assert_called_once()
          manager.shutdown.assert_called_once()
          worker.join.assert_called_once()
          queue.close.assert_called_once()

  def test_cleanup_logs_escalation_and_resource_shutdown(self) -> None:
    """Shutdown releases resources in dependency order and warns on escalation."""

    process, link, manager, queue, worker = self.setup_cleanup_resources()
    logger = Block.logger
    operations = Mock()
    for name, operation in (('terminate', process.terminate), ('kill', process.kill),
                             ('close_link', link.close), ('shutdown', manager.shutdown),
                             ('join_logger', worker.join), ('close_queue', queue.close)):
      operations.attach_mock(operation, name)
    with patch.object(block_module.connection, 'wait', return_value=[1]):
      Block._cleanup()
    completed = [call[0] for call in operations.mock_calls]
    for name in ('terminate', 'kill', 'close_link', 'shutdown', 'join_logger', 'close_queue'):
      self.assertIn(name, completed)
    self.assertLess(completed.index('kill'), completed.index('close_link'))
    self.assertLess(completed.index('close_link'), completed.index('shutdown'))
    self.assertLess(completed.index('join_logger'), completed.index('close_queue'))
    self.assertTrue(any(call.kwargs['level'] >= logging.WARNING
                        for call in logger.log.call_args_list))

  def test_cleanup_progress_logging_failure_does_not_skip_resources(self) -> None:
    """A logger that keeps failing still allows every resource owner to close."""

    for error_type in (OSError, KeyboardInterrupt):
      with self.subTest(error=error_type.__name__):
        process, link, manager, queue, worker = self.setup_cleanup_resources()
        error = error_type('progress logging failure')
        Block.no_raise = True
        with (patch.object(Block.logger, 'log', side_effect=error),
              patch.object(block_module.connection, 'wait', return_value=[1])):
          Block._cleanup()
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        link.close.assert_called_once()
        manager.shutdown.assert_called_once()
        worker.join.assert_called_once()
        queue.close.assert_called_once()
        self.assertIn(error, [actual for _, actual in
                              Block.run_failure_manager.snapshot()])

  def test_partial_startup_without_stop_event_still_releases_resources(
      self) -> None:
    """A missing Event is normal before synchronization objects are allocated."""

    process, link, manager, queue, worker = self.setup_cleanup_resources()
    Block.stop_event = None
    with patch.object(block_module.connection, 'wait', return_value=[1]):
      Block._cleanup()
    process.kill.assert_called_once_with()
    link.close.assert_called_once_with(manager=Block.run_failure_manager)
    manager.shutdown.assert_called_once_with()
    worker.join.assert_called_once_with(timeout=1.0)
    queue.close.assert_called_once_with()

  def test_finish_announcement_failure_cannot_skip_finish(self) -> None:
    """Logging after failed preparation cannot skip finish or connection release."""

    for error_type in (OSError, KeyboardInterrupt):
      with self.subTest(error=error_type.__name__):
        block = TestBlock()
        block._stop_event = Event()
        prepared = []
        error = error_type('finish logging failure')

        def fail_prepare():
          prepared.append(True)
          raise ValueError('prepare failure')

        def fail_log(level, msg):
          if prepared:
            raise error

        with (patch.object(block, '_set_block_logger'),
              patch.object(block, 'log', side_effect=fail_log),
              patch.object(block, 'prepare', side_effect=fail_prepare),
              patch.object(block, 'finish') as finish,
              patch.object(block, '_close_config_connections') as close):
          block.run()
        finish.assert_called_once()
        close.assert_called()
        self.assertIn(error, self.failures(block))

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
    logger = Block.logger

    with patch.object(block_module.connection, 'wait', return_value=[0, 1]):
      with self.assertRaises(ExceptionGroup):
        Block._finalize_run()

    second.terminate.assert_called_once_with()
    first.kill.assert_called_once_with()
    manager.shutdown.assert_called_once_with()
    worker.join.assert_called_once_with(timeout=1.0)
    queue.cancel_join_thread.assert_not_called()
    queue.close.assert_called_once_with()
    self.assertTrue(logger.error.call_count)
    self.assertGreaterEqual(logger.error.call_count, 3)
    self.assertEqual(Block._run_blocks, ())
    self.assertIsNone(Block.shared_mgr)

  def test_stop_and_queue_failures_still_run_other_cleanup(self) -> None:
    """Failure to signal stop or close a Queue cannot skip later steps."""

    self.setup_cleanup()
    manager, queue = Mock(), Mock()
    Block.stop_event = Mock()
    Block.stop_event.set.side_effect = KeyboardInterrupt()
    queue.close.side_effect = OSError('close')
    Block.shared_mgr, Block.log_queue = manager, queue
    Block.no_raise = True
    Block._cleanup()
    manager.shutdown.assert_called_once_with()
    queue.close.assert_called_once_with()
    self.assertEqual([type(error) for _, error in
                      Block.run_failure_manager.snapshot()],
                     [KeyboardInterrupt, OSError])

  def test_cleanup_closes_each_link_once_and_still_stops_manager(self) -> None:
    """A Link appearing on both Blocks is cleaned up once, independently."""

    self.setup_cleanup()
    source, target = TestBlock(), TestBlock()
    link = Link(source, target)
    self.addCleanup(link.close, Block.run_failure_manager)
    manager = Mock()
    Block.shared_mgr = manager
    Block._run_blocks = (source, target)
    with patch.object(link, 'close', side_effect=OSError('link')) as close:
      Block._cleanup()
    close.assert_called_once_with(manager=Block.run_failure_manager)
    manager.shutdown.assert_called_once_with()
    self.assertTrue(Block.run_failure_manager.snapshot())

  def test_reporting_failure_still_resets_runtime(self) -> None:
    """An unavailable error reporter must not prevent the final reset."""

    self.setup_cleanup()
    manager, queue = Mock(), Mock()
    Block.shared_mgr, Block.log_queue = manager, queue
    manager.shutdown.side_effect = RuntimeError('shutdown')
    error = OSError('reporting')
    Block.logger.error.side_effect = error
    with self.assertRaises(ExceptionGroup) as caught:
      Block._finalize_run()
    self.assertIn(error, exception_leaves(caught.exception))
    queue.close.assert_called_once_with()
    self.assertIsNone(Block.shared_mgr)
    self.assertIsNone(Block.log_queue)
    self.assertEqual(Block._run_blocks, ())

  def test_stop_event_failure_cannot_skip_finish(self) -> None:
    """The per-Block cleanup hook runs even when signaling stop fails."""

    block = TestBlock()
    block._stop_event = Mock()
    block._stop_event.set.side_effect = OSError('stop event')
    with (patch.object(block, '_set_block_logger'),
          patch.object(block, 'prepare', side_effect=ValueError('prepare')),
          patch.object(block, 'finish') as finish):
      block.run()
    finish.assert_called_once_with()
    self.assertTrue(self.failures(block))

  def test_inherited_pipe_failure_does_not_skip_peers_or_final_retry(
      self) -> None:
    """Failed inherited endpoints remain available until final cleanup."""

    block = TestBlock()
    block._stop_event = Event()
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
