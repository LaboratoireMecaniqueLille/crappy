# coding: utf-8

from crappy import Block
from crappy.blocks.meta_block import block as block_module
from multiprocessing import synchronize
from threading import Thread
from queue import Empty
from time import monotonic, sleep
from platform import system
import logging
import unittest
import signal
from unittest.mock import Mock, patch
from weakref import WeakSet

from tests.support.block_test_case import BlockTestBase
from tests.support.blocks import TestBlock
from tests.support.processes import stop_thread as join_thread


class TestBlockNoResponse(TestBlock):
  """Test Block that deliberately ignores the stop sequence for a while."""

  def loop(self) -> None:
    """Sleeps long enough for the cleanup code to have to intervene."""

    self.looped.set()
    sleep(30)


class TestBlockRaise(TestBlock):
  """Test Block raising after a peer has entered its unresponsive loop."""

  def __init__(self, wait_for_loop: synchronize.Event) -> None:
    super().__init__()
    self._wait_for_loop = wait_for_loop

  def loop(self) -> None:
    """Waits for the peer's loop to start before raising."""

    if not self._wait_for_loop.wait(3.0):
      raise RuntimeError("The peer Block never entered its loop")
    raise ValueError


class TestBlockIgnoreTerminate(TestBlockNoResponse):
  """POSIX child deliberately ignoring SIGTERM to require kill escalation."""

  def prepare(self) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    super().prepare()


class TestStartupSequence(BlockTestBase):
  def test_prepare_all_prepared(self) -> None:
    """Tests that calling prepare_all twice fails cleanly."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    # Manually break the current startup context before attempting the second
    # call, otherwise the first prepared Block would still be waiting
    Block.ready_barrier.abort()
    Block.thread_stop = True

    self.assertTrue(self._block.stop_event.wait(3.0))

    with self.assertRaises(ExceptionGroup):
      Block.prepare_all(log_level=logging.CRITICAL)

  def test_prepare_all_launched(self) -> None:
    """Tests that prepare_all refuses an inconsistent launched state."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    Block.ready_barrier.abort()
    Block.thread_stop = True

    self.assertTrue(self._block.stop_event.wait(3.0))

    Block.prepared_all = False
    Block.launched_all = True

    with self.assertRaises(ExceptionGroup):
      Block.prepare_all(log_level=logging.CRITICAL)

  def test_prepare_all_setup(self):
    """Preparation runs prepare; user work starts only after launch."""

    self._block = TestBlock()
    Block.prepare_all(log_level=logging.CRITICAL)
    self.assertTrue(self._block.prepared.wait(3.0))
    self.assertTrue(self._block.is_alive())
    self.assertFalse(self._block.begun.is_set())
    self.assertFalse(self._block.looped.is_set())
    Block.launch_all()
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())
    self.assertFalse(self._block.is_alive())
    self.assertEqual(self._block.exitcode, 0)

  @unittest.skipIf(system() not in ('Linux', 'Darwin'),
                   "Test irrelevant on Windows")
  def test_renice_all(self) -> None:
    """Tests that renice_all applies the requested niceness."""

    self._block = TestBlock()
    self._block.niceness = 5

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    # The unit under test is command construction. Mocking the system call
    # avoids depending on external ``ps``/``renice`` executables or permissions.
    with patch.object(Block, 'instances', WeakSet()), \
         patch.object(block_module.subprocess, 'call') as call:
      Block.renice_all(allow_root=False)

    call.assert_called_once_with(
        ['renice', '5', '-p', str(self._block.pid)],
        stdout=block_module.subprocess.DEVNULL)

    Block.ready_barrier.abort()
    Block.thread_stop = True

    self.assertTrue(self._block._stop_event.wait(3.0))

    Block.reset()

  def test_renice_all_not_prepared(self) -> None:
    """Tests that renice_all cannot be called before prepare."""

    self._block = TestBlock()

    with self.assertRaises(ExceptionGroup):
      Block.renice_all(allow_root=False)

    Block.reset()

  def test_renice_all_launched(self) -> None:
    """Tests that renice_all aborts on an inconsistent launched flag."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    Block.ready_barrier.abort()
    Block.thread_stop = True

    self.assertTrue(self._block.stop_event.wait(3.0))

    Block.launched_all = True

    with self.assertRaises(ExceptionGroup):
      Block.renice_all(allow_root=False)

  def test_launch_all(self) -> None:
    """Tests the normal end-to-end startup sequence."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    Block.launch_all()

    self.assertFalse(self._block.is_alive())

  def test_launch_all_no_prepared(self) -> None:
    """Tests that launch_all refuses to run before prepare."""

    self._block = TestBlock()

    with self.assertRaises(ExceptionGroup):
      Block.launch_all()

    Block.reset()

  def test_launch_all_launched(self) -> None:
    """Tests that launch_all aborts on a duplicated launch."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    Block.launched_all = True

    with self.assertRaises(ExceptionGroup):
      Block.launch_all()

  def test_stop_all(self) -> None:
    """Tests that stop_all can stop a running Crappy session."""

    def stop():
      """Stops the running session after a short delay."""

      if self._block.looped.wait(3.0):
        Block.stop_all()

    stop_thread = Thread(target=stop)
    self.addCleanup(join_thread, stop_thread, lambda: None)

    self._block = TestBlock(stop=False)

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    stop_thread.start()

    Block.launch_all()

    self.assertFalse(self._block.is_alive())

  def test_restart(self) -> None:
    """Tests that a fresh Crappy session can start after a completed one."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    Block.launch_all()

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    Block.launch_all()

  def test_log_thread_stops_after_reset(self) -> None:
    """A logging worker must stop when reset discards its startup context."""

    queue = Mock()

    def reset_while_reading(*args, **kwargs):
      Block.reset()
      raise Empty

    queue.get.side_effect = reset_while_reading
    Block._ensure_failure_manager()
    Block.log_queue = queue
    Block._log_target(queue)

    self.assertTrue(queue.get.called)
    self.assertTrue(queue.get.call_args.kwargs['block'])
    self.assertGreater(queue.get.call_args.kwargs['timeout'], 0)

  def test_block_not_responding(self) -> None:
    """Tests that cleanup terminates a Block that does not stop by itself."""

    self._block = TestBlockNoResponse()
    peer = TestBlockRaise(self._block.looped)
    blocks = (self._block, peer)

    def reap_blocks() -> None:
      # Keep process references across Block.reset, including assertion failures.
      for inst in blocks:
        if inst.is_alive():
          inst.kill()
        if inst.pid is not None:
          inst.join(1.0)

    self.addCleanup(reap_blocks)
    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    # Exercise the ten-second grace period without waiting that long
    # in wall-clock time.
    cleanup_start = monotonic()

    def accelerated_time() -> float:
      return cleanup_start + 100 * (monotonic() - cleanup_start)

    real_wait = block_module.connection.wait

    def accelerated_wait(objects, timeout=None):
      # Only accelerate the main grace-period wait, not Process.join internals.
      scaled = (timeout / 100
                if timeout is not None and isinstance(objects, dict) else timeout)
      return real_wait(objects, timeout=scaled)

    with patch.object(block_module, 'monotonic',
                      side_effect=accelerated_time), \
         patch.object(block_module.connection, 'wait',
                      side_effect=accelerated_wait), \
         patch.object(self._block, 'terminate',
                      wraps=self._block.terminate) as terminate:
      with self.assertRaises(ExceptionGroup):
        Block.launch_all()

      terminate.assert_called_once()

    self.assertTrue(self._block.looped.is_set())
    for inst in blocks:
      inst.join(1.0)
      self.assertFalse(inst.is_alive())

  @unittest.skipIf(system() == 'Windows', 'SIGTERM handling is POSIX-only')
  def test_cleanup_kills_block_ignoring_terminate(self) -> None:
    """A real SIGTERM-ignoring child is killed and reaped before reset."""

    self._block = TestBlockIgnoreTerminate()
    peer = TestBlockRaise(self._block.looped)
    blocks = (self._block, peer)

    def reap_blocks():
      for inst in blocks:
        if inst.is_alive():
          inst.kill()
        if inst.pid is not None:
          inst.join(timeout=1.0)

    self.addCleanup(reap_blocks)
    Block.prepare_all(log_level=logging.CRITICAL)
    self.assertTrue(self._block.prepared.wait(timeout=3.0))
    cleanup_start = monotonic()
    real_wait = block_module.connection.wait

    def accelerated_time():
      return cleanup_start + 100 * (monotonic() - cleanup_start)

    def accelerated_wait(objects, timeout=None):
      # Process.join must retain a real timeout long enough to reap the child.
      scaled = (timeout / 100
                if timeout is not None and isinstance(objects, dict) else timeout)
      return real_wait(objects, timeout=scaled)

    with (patch.object(block_module, 'monotonic', side_effect=accelerated_time),
          patch.object(block_module.connection, 'wait',
                       side_effect=accelerated_wait),
          patch.object(self._block, 'terminate',
                       wraps=self._block.terminate) as terminate,
          patch.object(self._block, 'kill', wraps=self._block.kill) as kill,
          self.assertRaises(ExceptionGroup)):
      Block.launch_all()

    terminate.assert_called_once()
    kill.assert_called_once()
    self.assertFalse(self._block.is_alive())
    self.assertEqual(self._block.exitcode, -signal.SIGKILL)
