# coding: utf-8

from crappy import Block
from crappy.blocks.meta_block import block as block_module
from multiprocessing import Barrier, Event, Value, Queue
from threading import Event as ThreadEvent, Thread
from types import SimpleNamespace
import logging
from unittest.mock import Mock, patch

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
    # The first loop records zero; main advances the counter afterward.
    self.assertTrue(self.wait_until(lambda: self._block.n_loops.value > 0),
                    "frequency bookkeeping did not advance")

    t = self._block.last_t.value
    n_l = self._block.loops.value

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertFalse(self._block.finished.is_set())

    # Multiple loops can share a clock tick, so wait for timing to advance too.
    self.assertTrue(self.wait_until(
      lambda: self._block.loops.value > n_l and
              self._block.last_t.value > t))
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
    """Frequency reporting uses the start time and counter supplied by main."""

    self._block = TestBlock()
    self._block.freq = 20.0
    self._block.display_freq = True
    self._block._last_t = 3.0
    self._block._last_fps = 0.0
    self._block._n_loops = 4
    stop = Mock()
    stop.wait.return_value = False
    self._block._stop_event = stop

    with patch.object(block_module, 'monotonic', return_value=3.02), \
         patch.object(self._block, 'log') as log:
      self._block._handle_freq()

    stop.wait.assert_called_once()
    self.assertAlmostEqual(stop.wait.call_args.args[0], 0.03)
    # _handle_freq must not replace the iteration's start timestamp.
    self.assertEqual(self._block._last_t, 3.0)
    self.assertEqual(self._block._last_fps, 3.0)
    self.assertEqual(self._block._n_loops, 0)
    log.assert_called_once_with(logging.INFO, f'loops/s: {4 / 3}')

  def _assert_loop_timing(self,
                          freq: float | None,
                          work_times: tuple[float, ...],
                          expected_starts: tuple[float, ...],
                          expected_waits: tuple[float, ...]) -> None:
    """Runs main with a simulated clock and known loop execution times."""

    self._block = TestBlock(stop=False)
    self._block.freq = freq
    self._block._last_fps = 0.0
    self._block._stop_event = Mock()
    self._block._pause_event = Mock()
    self._block._pause_event.is_set.return_value = False
    starts = list()
    waits = list()
    clock = SimpleNamespace(now=0.0)
    self._block._stop_event.is_set.side_effect = (
        lambda: len(starts) >= len(work_times))

    def do_work() -> None:
      # Check the timestamp before advancing time to the end of the work.
      self.assertAlmostEqual(self._block._last_t, clock.now)
      self.assertEqual(self._block._n_loops, len(starts))
      starts.append(clock.now)
      clock.now += work_times[len(starts) - 1]

    def wait_for_stop(timeout: float) -> bool:
      waits.append(timeout)
      if len(starts) >= len(work_times):
        return True
      clock.now += timeout
      return False

    self._block._stop_event.wait.side_effect = wait_for_stop
    with patch.object(block_module, 'monotonic',
                      side_effect=lambda: clock.now), \
         patch.object(self._block, 'loop', side_effect=do_work):
      self._block.main()

    self.assertEqual(len(starts), len(expected_starts))
    for actual, expected in zip(starts, expected_starts):
      self.assertAlmostEqual(actual, expected)
    self.assertEqual(len(waits), len(expected_waits))
    for actual, expected in zip(waits, expected_waits):
      self.assertAlmostEqual(actual, expected)
    self.assertEqual(self._block._n_loops, len(work_times))
    self.assertAlmostEqual(self._block._last_t, starts[-1])

  def test_frequency_includes_loop_work(self) -> None:
    """At 20 Hz, 20 ms of work leaves only 30 ms to wait."""

    self._assert_loop_timing(20.0, (0.02,) * 4,
                            (0.0, 0.05, 0.10, 0.15), (0.03,) * 4)

  def test_frequency_overrun_does_not_add_a_wait(self) -> None:
    """Work exceeding the 50 ms period must not incur another full period."""

    self._assert_loop_timing(20.0, (0.07,) * 4,
                            (0.0, 0.07, 0.14, 0.21), (0.0,) * 4)

  def test_frequency_does_not_catch_up_after_overrun(self) -> None:
    """An overrun must not shorten the interval following the next start."""

    self._assert_loop_timing(20.0, (0.07, 0.0, 0.01, 0.02),
                            (0.0, 0.07, 0.12, 0.17),
                            (0.0, 0.05, 0.04, 0.03))

  def test_no_frequency_does_not_wait(self) -> None:
    """With freq=None, main only advances time by the work performed."""

    self._assert_loop_timing(None, (0.02,) * 4,
                            (0.0, 0.02, 0.04, 0.06), ())
    self._block._stop_event.wait.assert_not_called()

  def test_paused_opportunities_obey_frequency(self) -> None:
    """A paused Block still spaces and counts its opportunities to run."""

    self._block = TestBlock(stop=False)
    self._block.freq = 20.0
    self._block._stop_event = Mock()
    self._block._pause_event = Mock()
    self._block._pause_event.is_set.return_value = True
    starts = list()
    clock = SimpleNamespace(now=0.0)
    self._block._stop_event.is_set.side_effect = lambda: len(starts) >= 3

    def wait_for_stop(timeout: float) -> bool:
      self.assertAlmostEqual(timeout, 0.05)
      starts.append(self._block._last_t)
      if len(starts) >= 3:
        return True
      clock.now += timeout
      return False

    self._block._stop_event.wait.side_effect = wait_for_stop
    with patch.object(block_module, 'monotonic',
                      side_effect=lambda: clock.now), \
         patch.object(self._block, 'loop') as loop:
      self._block.main()

    loop.assert_not_called()
    self.assertEqual(len(starts), 3)
    for actual, expected in zip(starts, (0.0, 0.05, 0.10)):
      self.assertAlmostEqual(actual, expected)
    self.assertEqual(self._block._n_loops, 3)

  def test_frequency_wait_is_interruptible(self) -> None:
    """A stop request wakes a long frequency wait without waiting its period."""

    self._block = TestBlock()
    self._block.freq = 0.1
    self._block.display_freq = True
    self._block._last_t = 0.0
    self._block._last_fps = -10.0
    self._block._n_loops = 4
    stop = Event()
    self._block._stop_event = stop
    waiting = ThreadEvent()
    errors = list()
    real_wait = stop.wait

    def notify_wait(timeout: float) -> bool:
      waiting.set()
      return real_wait(timeout)

    def handle_freq() -> None:
      try:
        self._block._handle_freq()
      except BaseException as error:
        errors.append(error)

    worker = Thread(target=handle_freq, daemon=True)
    with patch.object(block_module, 'monotonic', return_value=0.0), \
         patch.object(stop, 'wait', side_effect=notify_wait) as wait, \
         patch.object(self._block, 'log') as log:
      worker.start()
      try:
        self.assertTrue(waiting.wait(1.0))
        stop.set()
        worker.join(1.0)
        self.assertFalse(worker.is_alive())
      finally:
        stop.set()
        worker.join(1.0)

      wait.assert_called_once_with(10.0)
      log.assert_not_called()
    self.assertEqual(errors, [])
    self.assertEqual(self._block._n_loops, 4)
    self.assertEqual(self._block._last_fps, -10.0)
