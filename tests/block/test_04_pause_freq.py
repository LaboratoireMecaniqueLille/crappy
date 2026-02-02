# coding: utf-8

from crappy import Block
from multiprocessing import Barrier, Event, Value, Queue
from time import sleep

from .block_test_base import BlockTestBase, TestBlock


class TestPauseFreq(BlockTestBase):
  """"""

  def test_stop_event(self) -> None:
    """"""

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

    self.assertGreater(self._block.last_t.value, -1.0)
    self.assertGreater(self._block.last_fps.value, -1.0)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertEqual(self._block.n_loops.value, 0)
    self.assertEqual(self._block.loops.value, 0)

    Block.reset()

  def test_free_run(self) -> None:
    """"""

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

    self._block.join(0.5)

    self.assertGreater(self._block.last_t.value, -1.0)
    self.assertGreater(self._block.last_fps.value, -1.0)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertGreater(self._block.n_loops.value, 0)

    t = self._block.last_t.value
    n = self._block.n_loops.value
    n_l = self._block.loops.value

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertFalse(self._block.finished.is_set())

    self._block.join(0.5)

    self.assertGreater(self._block.last_t.value, t)
    self.assertGreater(self._block.n_loops.value, n)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertGreater(self._block.loops.value, n_l)

    stop.set()

    self._block.join(0.5)

    self.assertTrue(self._block.finished.is_set())

    Block.reset()

  def test_pause(self) -> None:
    """"""

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

    self._block.join(0.5)

    pause.set()
    sleep(0.5)

    self.assertGreater(self._block.last_t.value, -1.0)
    self.assertGreater(self._block.last_fps.value, -1.0)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertGreater(self._block.n_loops.value, 0)

    t = self._block.last_t.value
    n = self._block.n_loops.value
    n_l = self._block.loops.value

    self._block.join(0.5)

    stop.set()

    self._block.join(0.5)

    self.assertGreater(self._block.last_t.value, t)
    self.assertGreater(self._block.n_loops.value, n)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertEqual(self._block.loops.value, n_l)

    self.assertTrue(self._block.finished.is_set())

    Block.reset()

  def test_pause_resume(self) -> None:
    """"""

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

    self._block.join(0.5)

    pause.set()
    sleep(0.5)

    self.assertGreater(self._block.last_t.value, -1.0)
    self.assertGreater(self._block.last_fps.value, -1.0)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertGreater(self._block.n_loops.value, 0)

    t = self._block.last_t.value
    n_l = self._block.loops.value

    self._block.join(0.5)

    self.assertEqual(self._block.loops.value, n_l)

    pause.clear()

    self._block.join(0.5)

    self.assertGreater(self._block.last_t.value, t)
    self.assertGreater(self._block.loops.value, n_l)

    stop.set()

    self._block.join(0.5)

    self.assertTrue(self._block.finished.is_set())

    Block.reset()

  def test_start_pause(self) -> None:
    """"""

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

    self._block.join(0.5)

    self.assertEqual(self._block.loops.value, 0)

    t = self._block.last_t.value
    n = self._block.n_loops.value

    self._block.join(0.5)

    stop.set()

    self._block.join(0.5)

    self.assertGreater(self._block.last_t.value, t)
    self.assertGreater(self._block.n_loops.value, n)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertEqual(self._block.loops.value, 0)

    self.assertTrue(self._block.finished.is_set())

    Block.reset()

  def test_non_pausable(self) -> None:
    """"""

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

    self._block.join(0.5)

    self.assertGreater(self._block.last_t.value, -1.0)
    self.assertGreater(self._block.n_loops.value, -1.0)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertGreater(self._block.n_loops.value, 0)

    stop.set()

    self._block.join(0.5)

    self.assertTrue(self._block.finished.is_set())

    Block.reset()

  def test_handle_freq(self) -> None:
    """"""

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

    self._block.join(1.8)

    self.assertTrue(self._block._start_event.is_set())
    self.assertFalse(self._block._stop_event.is_set())
    self.assertFalse(self._block._ready_barrier.broken)
    self.assertFalse(self._block._raise_event.is_set())
    self.assertFalse(self._block._kbi_event.is_set())

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertFalse(self._block.finished.is_set())

    self.assertGreater(self._block.last_t.value, -1.0)
    self.assertGreater(self._block.last_fps.value, -1.0)
    self.assertGreaterEqual(self._block.last_t.value,
                            self._block.last_fps.value)
    self.assertGreater(self._block.n_loops.value, 0)
    self.assertGreater(self._block.loops.value, 0)

    last_t = self._block.last_t.value
    last_fps = self._block.last_fps.value
    loops = self._block.loops.value

    self._block.join(0.5)

    self.assertGreater(self._block.loops.value, loops)
    self.assertGreater(self._block.last_fps.value, last_fps)
    self.assertGreater(self._block.last_t.value, last_t)

    stop.set()
    self._block.join(0.5)

    Block.reset()
