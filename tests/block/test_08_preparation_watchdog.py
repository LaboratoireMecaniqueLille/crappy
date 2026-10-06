# coding: utf-8

import logging
from multiprocessing import Barrier, Event
import os
from threading import Event as ThreadEvent, Thread
from types import SimpleNamespace
from unittest.mock import patch

from crappy import Block
from crappy._global import CrappyFail
from crappy.blocks.meta_block import block as block_module

from .block_test_base import BlockTestBase, TestBlock


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
          Block.prepare_all(log_level=logging.CRITICAL)
          self._block.join(3.0)
          self.assertEqual(self._block.exitcode, exit_code)
          self.assertTrue(peer.prepared.wait(3.0))

          if no_raise:
            self._launch_with_deadline(no_raise=True)
          else:
            with self.assertRaises(CrappyFail):
              self._launch_with_deadline()

          peer.join(1.0)
          self.assertFalse(peer.is_alive())
          self.assertTrue(peer.finished.is_set())
          self.assertFalse(peer.begun.is_set())
          self.assertFalse(self._block.begun.is_set())
          self.assertFalse(Block.prepared_all)

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
        with self.assertRaises(CrappyFail):
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

    with patch.object(block_module, 'Thread') as thread:
      watchdog = thread.return_value
      watchdog.ident = None
      watchdog.start.side_effect = RuntimeError('Cannot start watchdog')
      with self.assertRaises(CrappyFail):
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

    with patch.object(block_module, 'Thread') as thread:
      watchdog = thread.return_value
      watchdog.ident = 1
      watchdog.is_alive.return_value = True
      with self.assertRaises(CrappyFail):
        self._launch_with_deadline()

      watchdog.join.assert_called_once_with(1.0)

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

    with self.assertRaises(KeyboardInterrupt):
      self._launch_with_deadline()

  def test_abort_just_after_barrier_release_prevents_launch(self) -> None:
    """An abort racing with a successful main wait must still prevent begin."""

    self._block = TestBlock()
    Block.prepare_all(log_level=logging.CRITICAL)
    self.assertTrue(self._block.prepared.wait(3.0))
    barrier = Block.ready_barrier
    failure = Block.raise_event
    start = Block.start_event
    wait_for_blocks = barrier.wait

    def abort_after_release() -> int:
      result = wait_for_blocks()
      failure.set()
      barrier.abort()
      return result

    with patch.object(barrier, 'wait', side_effect=abort_after_release):
      with self.assertRaises(CrappyFail):
        self._launch_with_deadline()

    self.assertFalse(start.is_set())
    self.assertFalse(self._block.begun.is_set())

  def test_disarm_during_sentinel_wait_ignores_result(self) -> None:
    """An exit notification must not abort startup after disarming."""

    stop = ThreadEvent()
    barrier = Barrier(1)
    failure = Event()
    block = SimpleNamespace(sentinel=17)

    def stop_and_report_exit(*args, **kwargs) -> list[int]:
      stop.set()
      return [block.sentinel]

    with patch.object(block_module.connection, 'wait',
                      side_effect=stop_and_report_exit):
      Block._watchdog_target(stop, (block,), barrier, failure)

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
    block = SimpleNamespace(sentinel=17)

    with patch.object(block_module.connection, 'wait') as wait:
      Block._watchdog_target(stop, (block,), barrier, failure)

    wait.assert_not_called()
    self.assertFalse(failure.is_set())
    self.assertFalse(barrier.broken)

  def test_watchdog_error_aborts_barrier(self) -> None:
    """The main wait must also be released if the sentinel watcher fails."""

    stop = ThreadEvent()
    barrier = Barrier(1)
    failure = Event()
    block = SimpleNamespace(sentinel=17)

    with patch.object(block_module.connection, 'wait',
                      side_effect=OSError('Invalid sentinel')):
      Block._watchdog_target(stop, (block,), barrier, failure)

    self.assertTrue(failure.is_set())
    self.assertTrue(barrier.broken)
