# coding: utf-8

import logging
from multiprocessing import Barrier, Event
import os
from threading import Event as ThreadEvent, Thread, current_thread
from types import SimpleNamespace
from unittest.mock import Mock, patch

from crappy import Block
from crappy._failure import FailureManager
from crappy._exceptions import ProcessExitError
from crappy.blocks.meta_block import block as block_module

from tests.support.block_test_case import BlockTestBase
from tests.support.blocks import TestBlock
from tests.support.assertions import exception_leaves


class TestBlockExitPrepare(TestBlock):
  """Exits during preparation without running any exception handler."""

  def __init__(self, exit_code: int) -> None:
    super().__init__()
    self._prepare_exit_code = exit_code

  def prepare(self) -> None:
    super().prepare()
    os._exit(self._prepare_exit_code)


class TestBlockWaitPrepare(TestBlock):
  """Remains in preparation until the parent releases or kills it."""

  def __init__(self) -> None:
    super().__init__()
    self.release_prepare = Event()

  def prepare(self) -> None:
    super().prepare()
    self.release_prepare.wait()


class TestBlockInterruptPrepare(TestBlock):
  """Exercises the existing KeyboardInterrupt handling during preparation."""

  def prepare(self) -> None:
    super().prepare()
    raise KeyboardInterrupt


class TestPreparationWatchdog(BlockTestBase):
  """Checks process-death detection and the watchdog's shutdown boundary."""

  def _launch_with_deadline(self, no_raise: bool = False) -> None:
    """Breaks the Barrier if a regression would otherwise hang the test."""

    barrier = Block.ready_barrier
    completed = ThreadEvent()
    timed_out = ThreadEvent()

    def abort_if_hung() -> None:
      if not completed.wait(3.0):
        timed_out.set()
        barrier.abort()

    guard = Thread(target=abort_if_hung, daemon=True)
    guard.start()
    try:
      Block.launch_all(no_raise=no_raise)
    finally:
      completed.set()
      guard.join(1.0)
      self.assertFalse(timed_out.is_set(),
                       "Startup hung until the test aborted the Barrier")

  def test_hard_exit_before_launch(self) -> None:
    """Even exit code zero must fail startup and unblock the other Blocks."""

    for exit_code in (0, 7):
      for no_raise in (False, True):
        with self.subTest(exit_code=exit_code, no_raise=no_raise):
          self._block = TestBlockExitPrepare(exit_code)
          peer = TestBlock()
          try:
            Block.prepare_all(log_level=logging.CRITICAL)
            # Spawn may take longer to import the test module on CI. The exit
            # timeout starts only once the child has reached prepare().
            self.assertTrue(self._block.prepared.wait(15.0),
                            "The exiting Block did not reach prepare()")
            self._block.join(3.0)
            self.assertEqual(self._block.exitcode, exit_code)
            self.assertTrue(peer.prepared.wait(15.0),
                            "The peer Block did not reach prepare()")

            shared = SimpleNamespace(shutdown=Mock())
            Block.shared_mgr = shared
            with self.assertRaises(ExceptionGroup) as caught:
              self._launch_with_deadline(no_raise=no_raise)
            failure, = exception_leaves(caught.exception)
            self.assertIsInstance(failure, ProcessExitError)
            self.assertEqual(failure.process_name, self._block.name)
            shared.shutdown.assert_called_once_with()

            peer.join(1.0)
            self.assertFalse(peer.is_alive())
            self.assertTrue(peer.finished.is_set())
            self.assertFalse(peer.begun.is_set())
            self.assertFalse(self._block.begun.is_set())
            self.assertFalse(Block.prepared_all)
          finally:
            # subTest continues after failures; clean up before the next case.
            if Block.prepared_all:
              Block.no_raise = True
              Block._cleanup()
              Block.reset()

  def test_killed_while_watchdog_is_running(self) -> None:
    """A kill during preparation breaks the main and peer Barrier waits."""

    self._block = TestBlockWaitPrepare()
    peer = TestBlock()
    Block.prepare_all(log_level=logging.CRITICAL)
    self.assertTrue(self._block.prepared.wait(3.0))
    self.assertTrue(peer.prepared.wait(3.0))

    watchdog_running = ThreadEvent()
    watchdog_stopped = ThreadEvent()
    target = Block._watchdog_target

    def monitored_target(*args) -> None:
      watchdog_running.set()
      try:
        target(*args)
      finally:
        watchdog_stopped.set()

    def kill_during_prepare() -> None:
      if watchdog_running.wait(3.0):
        self._block.kill()

    killer = Thread(target=kill_during_prepare, daemon=True)
    killer.start()
    try:
      with patch.object(Block, '_watchdog_target',
                        side_effect=monitored_target):
        with self.assertRaises(ExceptionGroup):
          self._launch_with_deadline()
    finally:
      killer.join(3.0)

    self._block.join(1.0)
    peer.join(1.0)
    self.assertFalse(killer.is_alive())
    self.assertTrue(watchdog_stopped.is_set())
    self.assertIsNotNone(self._block.exitcode)
    self.assertNotEqual(self._block.exitcode, 0)
    self.assertFalse(peer.is_alive())
    self.assertFalse(peer.begun.is_set())

  def test_success_stops_watchdog(self) -> None:
    """Normal startup disarms the watchdog while all Blocks are still alive."""

    self._block = TestBlock()
    Block.prepare_all(log_level=logging.CRITICAL)
    self.assertTrue(self._block.prepared.wait(3.0))
    watchdog_stopped = ThreadEvent()
    target = Block._watchdog_target

    def monitored_target(*args) -> None:
      try:
        target(*args)
      finally:
        watchdog_stopped.set()

    with patch.object(Block, '_watchdog_target', side_effect=monitored_target):
      self._launch_with_deadline()

    self.assertTrue(watchdog_stopped.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertFalse(self._block.is_alive())

  def test_watchdog_start_failure_prevents_launch(self) -> None:
    """Failing to start the watchdog must abort and clean up startup."""

    self._block = TestBlock()
    Block.prepare_all(log_level=logging.CRITICAL)
    self.assertTrue(self._block.prepared.wait(3.0))
    start = Block.start_event

    watchdog = Mock(spec=Thread)
    watchdog.ident = None
    watchdog.start.side_effect = RuntimeError('Cannot start watchdog')

    def create_thread(*args, **kwargs):
      if kwargs.get('name') == 'crappy.barrier-watchdog':
        return watchdog
      return Thread(*args, **kwargs)

    with patch.object(block_module, 'Thread', side_effect=create_thread):
      with self.assertRaises(ExceptionGroup):
        self._launch_with_deadline()

      watchdog.join.assert_not_called()

    self._block.join(1.0)
    self.assertFalse(start.is_set())
    self.assertFalse(self._block.begun.is_set())
    self.assertFalse(self._block.is_alive())

  def test_watchdog_shutdown_failure_prevents_launch(self) -> None:
    """A watchdog still alive after join must prevent the common start."""

    self._block = TestBlock()
    Block.prepare_all(log_level=logging.CRITICAL)
    self.assertTrue(self._block.prepared.wait(3.0))
    start = Block.start_event

    watchdog = Mock(spec=Thread)
    watchdog.ident = 1
    watchdog.is_alive.return_value = True

    def create_thread(*args, **kwargs):
      if kwargs.get('name') == 'crappy.barrier-watchdog':
        return watchdog
      return Thread(*args, **kwargs)

    with patch.object(block_module, 'Thread', side_effect=create_thread):
      with self.assertRaises(ExceptionGroup):
        self._launch_with_deadline()

      watchdog.join.assert_called_once_with(timeout=0.5)

    self._block.join(1.0)
    self.assertFalse(start.is_set())
    self.assertFalse(self._block.begun.is_set())
    self.assertFalse(self._block.is_alive())

  def test_alive_block_can_keep_preparing_across_watchdog_timeouts(self) -> None:
    """Sentinel polling timeouts must not become preparation deadlines."""

    self._block = TestBlockWaitPrepare()
    Block.prepare_all(log_level=logging.CRITICAL)
    self.assertTrue(self._block.prepared.wait(3.0))
    start = Block.start_event
    watchdog_running = ThreadEvent()
    started_early = ThreadEvent()
    target = Block._watchdog_target

    def monitored_target(*args) -> None:
      watchdog_running.set()
      target(*args)

    def release_after_timeouts() -> None:
      if watchdog_running.wait(3.0):
        if start.wait(0.25):
          started_early.set()
        self._block.release_prepare.set()

    releaser = Thread(target=release_after_timeouts, daemon=True)
    releaser.start()
    try:
      with patch.object(Block, '_watchdog_target',
                        side_effect=monitored_target):
        self._launch_with_deadline()
    finally:
      releaser.join(3.0)

    self.assertFalse(releaser.is_alive())
    self.assertFalse(started_early.is_set())
    self.assertTrue(self._block.begun.is_set())

  def test_preparation_interrupt_remains_keyboard_interrupt(self) -> None:
    """A cooperative interrupt must not be reclassified as a hard exit."""

    self._block = TestBlockInterruptPrepare()
    Block.prepare_all(log_level=logging.CRITICAL)
    self._block.join(3.0)
    self.assertFalse(self._block.is_alive())

    with self.assertRaises(BaseExceptionGroup):
      self._launch_with_deadline()

  def test_abort_just_after_barrier_release_prevents_launch(self) -> None:
    """An abort racing with a successful main wait must still prevent begin."""

    self._block = TestBlock()
    Block.prepare_all(log_level=logging.CRITICAL)
    self.assertTrue(self._block.prepared.wait(3.0))
    barrier = Block.ready_barrier
    start = Block.start_event
    wait_for_blocks = barrier.wait

    def abort_after_release() -> int:
      result = wait_for_blocks()
      Block.run_failure_manager.capture_exception(ValueError('abort'))
      barrier.abort()
      return result

    with patch.object(barrier, 'wait', side_effect=abort_after_release):
      with self.assertRaises(ExceptionGroup):
        self._launch_with_deadline()

    self.assertFalse(start.is_set())
    self.assertFalse(self._block.begun.is_set())

  def test_disarm_during_sentinel_wait_ignores_result(self) -> None:
    """An exit notification must not abort startup after disarming."""

    stop = ThreadEvent()
    barrier = Barrier(1)
    failure = Event()
    block = SimpleNamespace(sentinel=17, name='test process')

    def stop_and_report_exit(*args, **kwargs) -> list[int]:
      stop.set()
      return [block.sentinel]

    with patch.object(block_module.connection, 'wait',
                      side_effect=stop_and_report_exit):
      Block.watchdog_stop, Block.ready_barrier = stop, barrier
      Block._run_blocks = (block,)
      Block.watchdog_thread = current_thread()
      Block.run_failure_manager = FailureManager('main', 'synchronize')
      Block.run_failure_manager._logger = Mock()
      Block._watchdog_target()

    self.assertFalse(failure.is_set())
    self.assertFalse(barrier.broken)

  def test_disarmed_watchdog_stays_stopped_during_cleanup(self) -> None:
    """The global stop Event must not reactivate a disarmed watchdog."""

    stop = ThreadEvent()
    stop.set()
    Block.stop_event = Event()
    Block.stop_event.set()
    barrier = Barrier(1)
    failure = Event()
    block = SimpleNamespace(sentinel=17, name='test process')

    with patch.object(block_module.connection, 'wait') as wait:
      Block.watchdog_stop, Block.ready_barrier = stop, barrier
      Block._run_blocks = (block,)
      Block.watchdog_thread = current_thread()
      Block.run_failure_manager = FailureManager('main', 'synchronize')
      Block.run_failure_manager._logger = Mock()
      Block._watchdog_target()

    wait.assert_not_called()
    self.assertFalse(failure.is_set())
    self.assertFalse(barrier.broken)

  def test_watchdog_error_aborts_barrier(self) -> None:
    """The main wait must also be released if the sentinel watcher fails."""

    stop = ThreadEvent()
    barrier = Barrier(1)
    failure = Event()
    block = SimpleNamespace(sentinel=17, name='test process')

    with patch.object(block_module.connection, 'wait',
                      side_effect=OSError('Invalid sentinel')):
      Block.watchdog_stop, Block.ready_barrier = stop, barrier
      Block._run_blocks = (block,)
      Block.watchdog_thread = current_thread()
      Block.run_failure_manager = FailureManager('main', 'synchronize')
      Block.run_failure_manager._logger = Mock()
      Block._watchdog_target()

    self.assertTrue(Block.run_failure_manager.snapshot())
    self.assertTrue(barrier.broken)
