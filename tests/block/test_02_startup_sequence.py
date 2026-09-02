# coding: utf-8

from crappy import Block
from crappy._global import CrappyFail
from crappy.blocks.meta_block import block as block_module
from multiprocessing import synchronize, queues, get_start_method, Event
from multiprocessing.sharedctypes import Synchronized
from threading import Thread
from time import monotonic, sleep
from platform import system
import logging
import unittest
from unittest.mock import patch

from .block_test_base import BlockTestBase, TestBlock


class TestBlockNoResponse(TestBlock):
  """Test Block that deliberately ignores the stop sequence for a while."""

  def loop(self) -> None:
    """Sleeps long enough for the cleanup code to have to intervene."""

    sleep(10)


class TestBlockRaise(TestBlock):
  """Test Block raising an exception from loop."""

  def loop(self) -> None:
    """Raises immediately when the main loop is entered."""

    raise ValueError


class TestStartupSequence(BlockTestBase):
  """Tests the class methods driving the startup and shutdown sequence.

  These tests cover the interactions between prepare_all, renice_all,
  launch_all and _cleanup.
  """

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

    with self.assertRaises(CrappyFail):
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

    with self.assertRaises(CrappyFail):
      Block.prepare_all(log_level=logging.CRITICAL)

  def test_prepare_all_setup(self) -> None:
    """Tests that prepare_all configures all shared objects properly."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    # The Block should have started and reached its prepare stage
    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(Block.prepared_all)
    self.assertFalse(Block.launched_all)

    # The shared synchronization objects should be instantiated and left in
    # their initial state
    self.assertIsInstance(Block.ready_barrier, synchronize.Barrier)
    self.assertEqual(Block.ready_barrier.parties, len(Block.instances) + 1)
    self.assertIsInstance(Block.shared_t0, Synchronized)
    self.assertEqual(Block.shared_t0.value, -1.0)

    for event in (Block.stop_event, Block.start_event, Block.pause_event,
                  Block.raise_event, Block.kbi_event):
      with self.subTest(event=event):
        self.assertIsNotNone(event)
        self.assertFalse(event.is_set())

    # Logging should also be configured at the class level
    self.assertIsInstance(Block.log_queue, queues.Queue)
    self.assertIsInstance(Block.log_thread, Thread)
    if get_start_method() == 'spawn':
      self.assertTrue(Block.log_thread.is_alive())
    else:
      self.assertFalse(Block.log_thread.is_alive())

    # The Block instance must reference the exact same shared objects
    for cls, inst in zip((Block.stop_event, Block.start_event,
                          Block.pause_event, Block.raise_event,
                          Block.kbi_event, Block.ready_barrier,
                          Block.shared_t0, Block.log_queue),
                         (self._block._stop_event, self._block._start_event,
                          self._block._pause_event, self._block._raise_event,
                          self._block._kbi_event, self._block._ready_barrier,
                          self._block._instance_t0, self._block._log_queue)):
      self.assertIs(inst, cls)

    self.assertTrue(self._block.is_alive())
    self.assertFalse(Block.thread_stop)

    # Abort the barrier to force the cleanup path
    Block.ready_barrier.abort()
    Block.thread_stop = True

    self.assertTrue(self._block.stop_event.wait(3.0))
    self.assertTrue(self.wait_until(lambda: not Block.log_thread.is_alive()))

    self.assertTrue(self._block.finished.wait(3.0))
    self._block.join(1.0)

    self.assertFalse(self._block.looped.is_set())
    self.assertFalse(self._block.begun.is_set())
    self.assertFalse(self._block.is_alive())

    Block.reset()

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
    with patch.object(block_module.subprocess, 'call') as call:
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

    with self.assertRaises(RuntimeError):
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

    with self.assertRaises(CrappyFail):
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

    with self.assertRaises(RuntimeError):
      Block.launch_all()

    Block.reset()

  def test_launch_all_launched(self) -> None:
    """Tests that launch_all aborts on a duplicated launch."""

    self._block = TestBlock()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    Block.launched_all = True

    with self.assertRaises(CrappyFail):
      Block.launch_all()

  def test_stop_all(self) -> None:
    """Tests that stop_all can stop a running Crappy session."""

    def stop():
      """Stops the running session after a short delay."""

      if self._block.looped.wait(3.0):
        Block.stop_all()

    stop_thread = Thread(target=stop)

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

  def test_cleanup(self) -> None:
    """Tests the different raise/no-raise combinations of _cleanup."""

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.no_raise = False
    Block._set_logger()

    Block._cleanup()

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.no_raise = False
    Block._set_logger()

    Block.raise_event.set()

    with self.assertRaises(CrappyFail):
      Block._cleanup()

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.no_raise = True
    Block._set_logger()

    Block.raise_event.set()

    Block._cleanup()

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.no_raise = False
    Block._set_logger()

    Block.kbi_event.set()

    with self.assertRaises(KeyboardInterrupt):
      Block._cleanup()

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.no_raise = True
    Block._set_logger()

    Block.kbi_event.set()

    Block._cleanup()

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.no_raise = False
    Block._set_logger()

    Block.raise_event.set()
    Block.kbi_event.set()

    with self.assertRaises(CrappyFail):
      Block._cleanup()

    Block.stop_event = Event()
    Block.raise_event = Event()
    Block.kbi_event = Event()
    Block.no_raise = True
    Block._set_logger()

    Block.raise_event.set()
    Block.kbi_event.set()

    Block._cleanup()

  def test_block_not_responding(self) -> None:
    """Tests that cleanup terminates a Block that does not stop by itself."""

    self._block = TestBlockNoResponse()
    _ = TestBlockRaise()

    Block.prepare_all(log_level=logging.CRITICAL)

    self.assertTrue(self._block.prepared.wait(3.0))

    # Exercise the three-second forced-termination path without making the
    # unit test spend three wall-clock seconds in it.
    cleanup_start = monotonic()

    def accelerated_time() -> float:
      return cleanup_start + 100 * (monotonic() - cleanup_start)

    with patch.object(block_module, 'time', side_effect=accelerated_time), \
         patch.object(block_module, 'sleep', side_effect=lambda _: sleep(0.01)):
      with self.assertRaises(CrappyFail):
        Block.launch_all()

    for inst in Block.instances:
      inst.join(1.0)
      self.assertFalse(inst.is_alive())
