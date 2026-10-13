# coding: utf-8

"""Run-wide retention, grouped finalization, and main-entry eligibility."""

import logging
import os
from multiprocessing import Barrier, Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

from crappy import Block
from crappy._exceptions import ProcessExitError
from crappy._failure import FailureManager
from crappy.blocks.meta_block import block as block_module
from crappy.links import link_graph

from .block_test_base import BlockTestBase, TestBlock, exception_leaves
from .test_13_phase_capture import TestBlockHookFailure


class TestBlockExitMain(TestBlock):
  """Exits abruptly after the base lifecycle marks entry into main."""

  def main(self) -> None:
    os._exit(7)


class TestBlockConcurrentFailure(TestBlock):
  """Raises only after both participating Blocks have entered their loops."""

  def __init__(self, barrier, error):
    super().__init__(stop=False)
    self.barrier = barrier
    self.error = error

  def loop(self):
    self.barrier.wait(timeout=5)
    raise self.error


class TestCoordinator(BlockTestBase):
  def setup_run(self, entered=(1,)) -> FailureManager:
    """Supplies a frozen participant set without starting child Processes."""

    blocks = tuple(TestBlock() for _ in entered)
    for block, value in zip(blocks, entered):
      block._main_entered.value = value
    Block._run_blocks = blocks
    Block.stop_event = Event()
    manager = FailureManager('main', 'main')
    manager._logger = Mock()
    Block.run_failure_manager = manager
    return manager

  def test_start_uses_private_operations_and_finalizes_once(self):
    origin = ValueError('launch')
    seen = []

    def prepare(level):
      Block._run_blocks = (self._block,)
      seen.append(Block.run_failure_manager)

    def renice(allow_root):
      seen.append(Block.run_failure_manager)

    def launch():
      seen.append(Block.run_failure_manager)
      raise origin

    self._block = TestBlock()
    with (patch.object(Block, '_prepare_all', side_effect=prepare),
          patch.object(Block, '_renice_all', side_effect=renice),
          patch.object(Block, '_launch_all', side_effect=launch),
          patch.object(Block, 'prepare_all') as public_prepare,
          patch.object(Block, 'renice_all') as public_renice,
          patch.object(Block, 'launch_all') as public_launch,
          patch.object(Block, 'reset', wraps=Block.reset) as reset):
      with self.assertRaises(ExceptionGroup) as caught:
        Block.start_all(log_level=None)
    self.assertTrue(all(manager is seen[0] for manager in seen))
    self.assertEqual(exception_leaves(caught.exception), (origin,))
    reset.assert_called_once_with()
    public_prepare.assert_not_called()
    public_renice.assert_not_called()
    public_launch.assert_not_called()
    self.assertIsNone(Block.run_failure_manager)

  def test_start_body_failure_skips_later_operations_and_finalizes_once(self):
    methods = ('_prepare_all', '_renice_all', '_launch_all')
    for failed in methods:
      with self.subTest(failed=failed):
        manager = self.setup_run((0,))
        origin = KeyboardInterrupt(failed)
        with (patch.object(Block, '_prepare_all') as prepare,
              patch.object(Block, '_renice_all') as renice,
              patch.object(Block, '_launch_all') as launch,
              patch.object(Block, 'reset', wraps=Block.reset) as reset):
          operations = (prepare, renice, launch)
          index = methods.index(failed)
          operations[index].side_effect = origin
          with self.assertRaises(BaseExceptionGroup) as caught:
            Block.start_all()
          for operation in operations[:index + 1]:
            self.assertEqual(operation.call_count, 1)
          for operation in operations[index + 1:]:
            operation.assert_not_called()
        self.assertEqual(exception_leaves(caught.exception), (origin,))
        manager._logger.error.assert_called_once()
        reset.assert_called_once_with()
        self.assertIsNone(Block.run_failure_manager)

  def test_start_no_raise_does_not_leak_a_pending_collection_group(self):
    manager = self.setup_run()
    origin, close = KeyboardInterrupt('launch'), OSError('queue close')
    queue = Mock()
    queue.close.side_effect = close
    Block.log_queue = queue
    with (patch.object(Block, '_prepare_all'),
          patch.object(Block, '_renice_all'),
          patch.object(Block, '_launch_all', side_effect=origin)):
      Block.start_all(no_raise=True)
    self.assertEqual([error for _, error in manager.snapshot()], [origin, close])
    self.assertEqual(manager._logger.error.call_count, 2)
    queue.close.assert_called_once_with()
    self.assertIsNone(Block.run_failure_manager)
    with self.assertRaises(BaseExceptionGroup) as caught:
      manager.raise_if_failed('retained')
    self.assertEqual(exception_leaves(caught.exception), (origin, close))
    self.assertEqual(manager._logger.error.call_count, 2)

  def test_manual_calls_keep_the_same_run_manager(self):
    self._block = TestBlock()
    Block.prepare_all(log_level=None)
    manager = Block.run_failure_manager
    with patch.object(block_module.subprocess, 'call', return_value=0):
      Block.renice_all(allow_root=False)
    self.assertIs(Block.run_failure_manager, manager)
    Block.launch_all()
    self.assertEqual(manager.snapshot(), ())
    self.assertIsNone(Block.run_failure_manager)

  def test_nonzero_renice_return_is_only_a_warning(self):
    self._block = TestBlock()
    Block.prepare_all(log_level=None)
    manager = Block.run_failure_manager
    with (patch.object(block_module, 'system', return_value='Linux'),
          patch.object(block_module.subprocess, 'call', return_value=1),
          patch.object(Block, 'cls_log') as log):
      Block.renice_all(allow_root=False)
    self.assertTrue(any(call.args[0] == logging.WARNING
                        for call in log.call_args_list))
    self.assertEqual(manager.snapshot(), ())
    self.assertIs(Block.run_failure_manager, manager)
    Block.launch_all()

  def test_renice_command_exception_is_retained_and_finalized(self):
    self._block = TestBlock()
    Block.prepare_all(log_level=None)
    manager = Block.run_failure_manager
    origin = FileNotFoundError('renice unavailable')
    with (patch.object(block_module, 'system', return_value='Linux'),
          patch.object(block_module.subprocess, 'call', side_effect=origin)):
      with self.assertRaises(ExceptionGroup) as caught:
        Block.renice_all(allow_root=False)
    self.assertEqual(exception_leaves(caught.exception), (origin,))
    self.assertEqual(manager.snapshot()[0][0].phase, 'renice')
    self.assertFalse(self._block.is_alive())
    self.assertIsNone(Block.run_failure_manager)

  def test_bootstrap_base_exceptions_are_grouped_despite_no_raise(self):
    for origin in (ValueError('setup'), KeyboardInterrupt('setup'),
                   SystemExit(3), BaseException('setup')):
      with self.subTest(error=type(origin)):
        self._block = TestBlock()
        with patch.object(Block, '_set_logger', side_effect=origin):
          with self.assertRaises(BaseExceptionGroup) as caught:
            Block.start_all(log_level=None, no_raise=True)
        self.assertEqual(exception_leaves(caught.exception), (origin,))
        self.assertIsNone(Block.run_failure_manager)
        self.assertFalse(Block.instances)

  def test_no_raise_requires_every_frozen_participant_and_nonempty_run(self):
    for entered, eligible in (((), False), ((0,), False), ((1, 0), False),
                              ((0, 1), False), ((1, 1), True)):
      with self.subTest(entered=entered):
        manager = self.setup_run(entered)
        origin = ValueError('origin')
        manager.capture_exception(origin)
        Block.no_raise = True
        if eligible:
          Block._finalize_run()
        else:
          with self.assertRaises(ExceptionGroup) as caught:
            Block._finalize_run()
          self.assertEqual(exception_leaves(caught.exception), (origin,))
        self.assertIsNone(Block.run_failure_manager)

  def test_no_raise_ignores_nonparticipants_and_suppresses_cleanup_interrupts(self):
    manager = self.setup_run()
    extra = TestBlock()
    self.assertEqual(extra._main_entered.value, 0)
    origin, close = ValueError('main'), KeyboardInterrupt('queue close')
    manager.capture_exception(origin)
    queue = Mock()
    queue.close.side_effect = close
    Block.log_queue = queue
    Block.no_raise = True
    Block._finalize_run()
    self.assertEqual([error for _, error in manager.snapshot()], [origin, close])
    queue.close.assert_called_once_with()
    self.assertFalse(Block.instances)

  def test_eligibility_is_captured_before_reset_and_handler_release(self):
    for entered in (0, 1):
      with self.subTest(entered=entered):
        manager = self.setup_run((entered,))
        flag = Block._run_blocks[0]._main_entered
        manager.capture_exception(ValueError('origin'))
        handler = Mock(spec=logging.Handler)
        handler.close.side_effect = lambda: setattr(flag, 'value', 1 - entered)
        Block.log_handlers = [handler]
        Block.no_raise = True
        if entered:
          Block._finalize_run()
        else:
          with self.assertRaises(ExceptionGroup):
            Block._finalize_run()
        self.assertEqual(flag.value, 1 - entered)
        handler.close.assert_called_once_with()

  def test_final_snapshot_contains_graph_flush_and_close_failures(self):
    manager = self.setup_run()
    origin, graph = ValueError('main'), OSError('graph reset')
    flush, close = RuntimeError('flush'), KeyboardInterrupt('close')
    manager.capture_exception(origin)
    first, second = Mock(spec=logging.Handler), Mock(spec=logging.Handler)
    first.flush.side_effect = flush
    first.close.side_effect = close
    Block.log_handlers = [first, second]
    with patch.object(link_graph, 'reset', side_effect=graph):
      with self.assertRaises(BaseExceptionGroup) as caught:
        Block._finalize_run()
    self.assertEqual(exception_leaves(caught.exception),
                     (origin, graph, flush, close))
    second.flush.assert_called_once_with()
    second.close.assert_called_once_with()
    self.assertIsNone(Block.logger)
    self.assertEqual(Block.log_handlers, [])
    self.assertIsNone(Block.run_failure_manager)
    self.assertEqual(manager._logger.error.call_count, 4)

  def test_groups_keep_owner_phase_order_and_original_group_structure(self):
    manager = self.setup_run()
    source = FailureManager('worker', 'main')
    source._logger = Mock()
    first, interrupt = ValueError('first'), KeyboardInterrupt('interrupt')
    nested = BaseExceptionGroup('original nested group', [first, interrupt])
    nested.add_note('original note')
    nested.__cause__ = OSError('original cause')
    first_report = source.capture_exception(nested)
    manager.forward(first_report, nested)
    local = RuntimeError('coordinator')
    manager.capture_exception(local)
    finish = OSError('finish')
    finish_report = source.capture_exception(finish, phase='finish')
    manager.forward(finish_report, finish)
    manager.forward(first_report, nested)
    second_main = TypeError('later main')
    later_report = source.capture_exception(second_main)
    manager.forward(later_report, second_main)
    logged = manager._logger.error.call_count
    with self.assertRaises(BaseExceptionGroup) as caught:
      Block._finalize_run()
    root = caught.exception
    self.assertEqual([group.message for group in root.exceptions],
                     ['worker', 'main'])
    worker = root.exceptions[0]
    self.assertEqual([group.message for group in worker.exceptions],
                     ['main', 'finish'])
    self.assertEqual(worker.exceptions[0].exceptions, (nested, second_main))
    self.assertIs(worker.exceptions[0].exceptions[0], nested)
    self.assertIn('original note', nested.__notes__)
    self.assertIsInstance(nested.__cause__, OSError)
    self.assertEqual(manager._logger.error.call_count, logged)

  def test_shared_manager_cleanup_uses_only_public_shutdown(self):
    manager = self.setup_run()
    shared = SimpleNamespace(shutdown=Mock())
    Block.shared_mgr = shared
    Block._finalize_run()
    self.assertEqual(manager.snapshot(), ())
    shared.shutdown.assert_called_once_with()

  def test_unreported_runtime_exit_is_synthesized_with_process_details(self):
    for no_raise in (False, True):
      with self.subTest(no_raise=no_raise):
        self._block = TestBlockExitMain()
        Block.prepare_all(log_level=None)
        shared = SimpleNamespace(shutdown=Mock())
        Block.shared_mgr = shared
        manager, flag = Block.run_failure_manager, self._block._main_entered
        if no_raise:
          Block.launch_all(no_raise=True)
        else:
          with self.assertRaises(ExceptionGroup) as caught:
            Block.launch_all()
          failure, = exception_leaves(caught.exception)
          self.assertIsInstance(failure, ProcessExitError)
          self.assertEqual(failure.process_name, self._block.name)
          self.assertEqual(failure.pid, self._block.pid)
          self.assertEqual(failure.exitcode, 7)
        self.assertEqual(flag.value, 1)
        self.assertEqual(self._block.exitcode, 7)
        self.assertEqual(len(manager.snapshot()), 1)
        shared.shutdown.assert_called_once_with()

  def test_single_reported_hook_failure_has_no_synthetic_duplicate(self):
    self._block = TestBlockHookFailure('main')
    Block.prepare_all(log_level=None)
    manager = Block.run_failure_manager
    with self.assertRaises(ExceptionGroup) as caught:
      Block.launch_all()
    failure, = exception_leaves(caught.exception)
    self.assertIsInstance(failure, ValueError)
    self.assertEqual(len(manager.snapshot()), 1)
    self.assertTrue(any('Original remote stack:' in note
                        for note in failure.__notes__))
    self.assertEqual(self._block.exitcode, 0)

  def test_concurrent_block_failures_preserve_interrupts_and_no_raise(self):
    for no_raise in (False, True):
      with self.subTest(no_raise=no_raise):
        barrier = Barrier(2)
        blocks = (TestBlockConcurrentFailure(barrier, ValueError('first')),
                  TestBlockConcurrentFailure(barrier, KeyboardInterrupt('second')))
        Block.prepare_all(log_level=None)
        manager = Block.run_failure_manager
        flags = tuple(block._main_entered for block in blocks)
        if no_raise:
          Block.launch_all(no_raise=True)
        else:
          with self.assertRaises(BaseExceptionGroup) as caught:
            Block.launch_all()
          self.assertEqual({type(error) for error in exception_leaves(
              caught.exception)}, {ValueError, KeyboardInterrupt})
          self.assertEqual({group.message for group in caught.exception.exceptions},
                           {block.name for block in blocks})
        self.assertEqual(tuple(flag.value for flag in flags), (1, 1))
        self.assertEqual(len(manager.snapshot()), 2)
        self.assertEqual({report.phase for report, _ in manager.snapshot()}, {'main'})
        self.assertTrue(all(block.exitcode == 0 for block in blocks))

  def test_stop_failure_is_retained_without_its_own_no_raise_policy(self):
    manager = self.setup_run()
    error = KeyboardInterrupt('stop signal')
    Block.stop_event = Mock()
    Block.stop_event.set.side_effect = error
    Block.no_raise = True
    with self.assertRaises(BaseExceptionGroup) as caught:
      Block.stop_all()
    self.assertEqual(exception_leaves(caught.exception), (error,))
    self.assertIs(manager.snapshot()[0][1], error)
    Block.stop_event = None
    Block._finalize_run()
