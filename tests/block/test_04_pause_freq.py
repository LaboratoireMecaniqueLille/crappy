# coding: utf-8

from crappy import Block
from crappy.blocks.meta_block import block as block_module
from multiprocessing import Barrier, Event, Value, Queue
import logging
from unittest.mock import patch

from .block_test_base import BlockTestBase, TestBlock


class TestPauseFreq(BlockTestBase):
  """Tests the pause handling and loop-frequency bookkeeping of Blocks."""

  def test_stop_event(self) -> None:
    """Tests that an already-set stop Event skips the loop body."""

    self._block = TestBlock()
    self._block.display_freq = True

    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = Event()
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = Event()
    self._block._instance_t0 = Value('d', 0.0)
    self._block._log_queue = Queue()

    self._block._start_event.set()
    self._block._stop_event.set()

    self._block.start()

    self._block.join(4.0)

    self.assertTrue(self._block._start_event.is_set())
    self.assertTrue(self._block._stop_event.is_set())
    self.assertFalse(self._block._ready_barrier.broken)
    self.assertFalse(self._block._raise_event.is_set())
    self.assertFalse(self._block._kbi_event.is_set())

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertFalse(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    # Timing bookkeeping is still initialized even though no user loop ran
    self.assertGreater(self._block.last_t.value, -1.0)
    self.assertGreater(self._block.last_fps.value, -1.0)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertEqual(self._block.n_loops.value, 0)
    self.assertEqual(self._block.loops.value, 0)

    Block.reset()

  def test_free_run(self) -> None:
    """Tests that an unpaused Block loops continuously until stopped."""

    self._block = TestBlock(stop=False)
    self._block.display_freq = True

    stop = Event()
    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = stop
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = Event()
    self._block._instance_t0 = Value('d', 0.0)
    self._block._log_queue = Queue()

    self._block._start_event.set()

    self._block.start()

    self.assertTrue(self._block.looped.wait(3.0))

    self.assertGreater(self._block.last_t.value, -1.0)
    self.assertGreater(self._block.last_fps.value, -1.0)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertGreater(self._block.n_loops.value, 0)

    t = self._block.last_t.value
    n_l = self._block.loops.value

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertFalse(self._block.finished.is_set())

    # The loop counters should keep increasing while the Block is running.
    self.assertTrue(self.wait_until(lambda: self._block.loops.value > n_l))
    self.assertGreater(self._block.last_t.value, t)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertGreater(self._block.loops.value, n_l)

    stop.set()

    self.assertTrue(self._block.finished.wait(3.0))

    Block.reset()

  def test_pause(self) -> None:
    """Tests that pausing stops calling loop but not freq handling."""

    self._block = TestBlock(stop=False)
    self._block.display_freq = True

    stop = Event()
    pause = Event()
    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = stop
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = pause
    self._block._instance_t0 = Value('d', 0.0)
    self._block._log_queue = Queue()

    self._block._start_event.set()

    self._block.start()

    self.assertTrue(self._block.looped.wait(3.0))

    pause.set()

    n_l = self.wait_until_stable(lambda: self._block.loops.value)

    self.assertGreater(self._block.last_t.value, -1.0)
    self.assertGreater(self._block.last_fps.value, -1.0)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertGreater(self._block.loops.value, 0)

    t = self._block.last_t.value

    stop.set()

    self.assertTrue(self._block.finished.wait(3.0))

    # While paused, the timing bookkeeping still advances but the actual user
    # loop count should remain constant.
    self.assertGreater(self._block.last_t.value, t)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertEqual(self._block.loops.value, n_l)

    Block.reset()

  def test_pause_resume(self) -> None:
    """Tests that clearing the pause Event resumes the normal looping."""

    self._block = TestBlock(stop=False)
    self._block.display_freq = True

    stop = Event()
    pause = Event()
    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = stop
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = pause
    self._block._instance_t0 = Value('d', 0.0)
    self._block._log_queue = Queue()

    self._block._start_event.set()

    self._block.start()

    self.assertTrue(self._block.looped.wait(3.0))

    pause.set()

    n_l = self.wait_until_stable(lambda: self._block.loops.value)

    self.assertGreater(self._block.last_t.value, -1.0)
    self.assertGreater(self._block.last_fps.value, -1.0)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertGreater(self._block.loops.value, 0)

    t = self._block.last_t.value
    self.assertEqual(self._block.loops.value, n_l)

    pause.clear()

    self.assertTrue(self.wait_until(lambda: self._block.loops.value > n_l))

    self.assertGreater(self._block.last_t.value, t)
    self.assertGreater(self._block.loops.value, n_l)

    stop.set()

    self.assertTrue(self._block.finished.wait(3.0))

    Block.reset()

  def test_start_pause(self) -> None:
    """Tests a Block that starts already paused."""

    self._block = TestBlock(stop=False)
    self._block.pausable = True
    self._block.display_freq = True

    stop = Event()
    pause = Event()
    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = stop
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = pause
    self._block._instance_t0 = Value('d', 0.0)
    self._block._log_queue = Queue()

    self._block._start_event.set()

    pause.set()

    self._block.start()

    self.assertTrue(self._block.begun.wait(3.0))
    self.wait_until_stable(lambda: self._block.loops.value)

    self.assertEqual(self._block.loops.value, 0)

    t = self._block.last_t.value

    stop.set()

    self.assertTrue(self._block.finished.wait(3.0))

    self.assertGreater(self._block.last_t.value, t)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertEqual(self._block.loops.value, 0)

    Block.reset()

  def test_non_pausable(self) -> None:
    """Tests that a non-pausable Block ignores the pause Event."""

    self._block = TestBlock(stop=False)
    self._block.pausable = False

    stop = Event()
    pause = Event()
    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = stop
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = pause
    self._block._instance_t0 = Value('d', 0.0)
    self._block._log_queue = Queue()

    self._block._start_event.set()

    pause.set()

    self._block.start()

    self.assertTrue(self._block.looped.wait(3.0))

    self.assertGreater(self._block.last_t.value, -1.0)
    self.assertGreater(self._block.n_loops.value, -1.0)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertGreater(self._block.loops.value, 0)

    stop.set()

    self.assertTrue(self._block.finished.wait(3.0))

    Block.reset()

  def test_handle_freq(self) -> None:
    """Tests frequency bookkeeping without waiting for wall-clock intervals."""

    self._block = TestBlock()
    self._block.display_freq = True
    self._block._last_t = 1.0
    self._block._last_fps = 0.0
    self._block._n_loops = 4

    with patch.object(block_module, 'time_ns', return_value=3_000_000_000), \
         patch.object(self._block, 'log') as log:
      self._block._handle_freq()

    self.assertEqual(self._block._last_t, 3.0)
    self.assertEqual(self._block._last_fps, 3.0)
    self.assertEqual(self._block._n_loops, 0)
    log.assert_called_once()
    self.assertEqual(log.call_args.args[0], logging.INFO)
