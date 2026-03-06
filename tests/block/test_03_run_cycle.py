# coding: utf-8

from crappy import Block
from multiprocessing import Barrier, Event, Value, Queue

from .block_test_base import BlockTestBase, TestBlock


class TestBlockRaisePrepare(TestBlock):
  """Test Block raising an exception from prepare."""

  def prepare(self) -> None:
    """Runs the normal prepare code and then raises."""

    super().prepare()
    raise ValueError


class TestBlockRaiseBegin(TestBlock):
  """Test Block raising an exception from begin."""

  def begin(self) -> None:
    """Runs the normal begin code and then raises."""

    super().begin()
    raise ValueError


class TestBlockRaiseLoop(TestBlock):
  """Test Block raising an exception from loop."""

  def loop(self) -> None:
    """Runs one normal loop iteration and then raises."""

    super().loop()
    raise ValueError


class TestBlockRaiseFinish(TestBlock):
  """Test Block raising an exception from finish."""

  def finish(self) -> None:
    """Runs the normal finish code and then raises."""

    super().finish()
    raise ValueError


class TestRunCycle(BlockTestBase):
  """Tests the per-Block execution cycle driven by Block.run."""

  def test_normal_run(self) -> None:
    """Tests the nominal prepare/begin/loop/finish sequence."""

    self._block = TestBlock()

    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = Event()
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = Event()
    self._block._instance_t0 = Value('d', 0.0)
    self._block._log_queue = Queue()

    self._block._start_event.set()

    self._block.start()

    self._block.join(4.0)

    # The Block should stop on its own without reporting any error.
    self.assertTrue(self._block._start_event.is_set())
    self.assertTrue(self._block._stop_event.is_set())
    self.assertFalse(self._block._ready_barrier.broken)
    self.assertFalse(self._block._raise_event.is_set())
    self.assertFalse(self._block._kbi_event.is_set())

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    self.assertGreater(self._block.last_t.value, -1.0)
    self.assertGreater(self._block.last_fps.value, -1.0)

    Block.reset()

  def test_raise_prepare(self) -> None:
    """Tests the behavior when prepare raises."""

    self._block = TestBlockRaisePrepare()

    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = Event()
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = Event()
    self._block._instance_t0 = Value('d', 0.0)
    self._block._log_queue = Queue()

    self._block.start()

    self._block.join(4.0)

    # An exception during prepare should break the barrier and propagate to the
    # shared raise Event.
    self.assertFalse(self._block._start_event.is_set())
    self.assertTrue(self._block._stop_event.is_set())
    self.assertTrue(self._block._ready_barrier.broken)
    self.assertTrue(self._block._raise_event.is_set())
    self.assertFalse(self._block._kbi_event.is_set())

    self.assertTrue(self._block.prepared.is_set())
    self.assertFalse(self._block.begun.is_set())
    self.assertFalse(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    self.assertEqual(self._block.last_t.value, -1.0)
    self.assertEqual(self._block.last_fps.value, -1.0)

    Block.reset()

  def test_raise_begin(self) -> None:
    """Tests the behavior when begin raises."""

    self._block = TestBlockRaiseBegin()

    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = Event()
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = Event()
    self._block._instance_t0 = Value('d', 0.0)
    self._block._log_queue = Queue()

    self._block._start_event.set()

    self._block.start()

    self._block.join(4.0)

    self.assertTrue(self._block._start_event.is_set())
    self.assertTrue(self._block._stop_event.is_set())
    self.assertFalse(self._block._ready_barrier.broken)
    self.assertTrue(self._block._raise_event.is_set())
    self.assertFalse(self._block._kbi_event.is_set())

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertFalse(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    self.assertEqual(self._block.last_t.value, -1.0)
    self.assertEqual(self._block.last_fps.value, -1.0)

    Block.reset()

  def test_raise_loop(self) -> None:
    """Tests the behavior when loop raises."""

    self._block = TestBlockRaiseLoop()

    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = Event()
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = Event()
    self._block._instance_t0 = Value('d', 0.0)
    self._block._log_queue = Queue()

    self._block._start_event.set()

    self._block.start()

    self._block.join(4.0)

    self.assertTrue(self._block._start_event.is_set())
    self.assertTrue(self._block._stop_event.is_set())
    self.assertFalse(self._block._ready_barrier.broken)
    self.assertTrue(self._block._raise_event.is_set())
    self.assertFalse(self._block._kbi_event.is_set())

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    self.assertGreater(self._block.last_t.value, -1.0)
    self.assertGreater(self._block.last_fps.value, -1.0)

    Block.reset()

  def test_raise_finish(self) -> None:
    """Tests the behavior when the Block ends with an exception path."""

    self._block = TestBlockRaiseLoop()

    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = Event()
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = Event()
    self._block._instance_t0 = Value('d', 0.0)
    self._block._log_queue = Queue()

    self._block._start_event.set()

    self._block.start()

    self._block.join(4.0)

    self.assertTrue(self._block._start_event.is_set())
    self.assertTrue(self._block._stop_event.is_set())
    self.assertFalse(self._block._ready_barrier.broken)
    self.assertTrue(self._block._raise_event.is_set())
    self.assertFalse(self._block._kbi_event.is_set())

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    self.assertGreater(self._block.last_t.value, -1.0)
    self.assertGreater(self._block.last_fps.value, -1.0)

    Block.reset()

  def test_brake_barrier(self) -> None:
    """Tests the behavior when the ready barrier is already broken."""

    self._block = TestBlock()

    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = Event()
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = Event()
    self._block._instance_t0 = Value('d', 0.0)
    self._block._log_queue = Queue()

    self._block._ready_barrier.abort()

    self._block.start()

    self._block.join(4.0)

    self.assertFalse(self._block._start_event.is_set())
    self.assertTrue(self._block._stop_event.is_set())
    self.assertTrue(self._block._ready_barrier.broken)
    self.assertFalse(self._block._raise_event.is_set())
    self.assertFalse(self._block._kbi_event.is_set())

    self.assertTrue(self._block.prepared.is_set())
    self.assertFalse(self._block.begun.is_set())
    self.assertFalse(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    self.assertEqual(self._block.last_t.value, -1.0)
    self.assertEqual(self._block.last_fps.value, -1.0)

    Block.reset()

  def test_no_start(self) -> None:
    """Tests the timeout path when the start Event is never set."""

    self._block = TestBlock()

    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = Event()
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = Event()
    self._block._instance_t0 = Value('d', 0.0)
    self._block._log_queue = Queue()

    self._block.start()

    self._block.join(4.0)

    self.assertFalse(self._block._start_event.is_set())
    self.assertTrue(self._block._stop_event.is_set())
    self.assertFalse(self._block._ready_barrier.broken)
    self.assertTrue(self._block._raise_event.is_set())
    self.assertFalse(self._block._kbi_event.is_set())

    self.assertTrue(self._block.prepared.is_set())
    self.assertFalse(self._block.begun.is_set())
    self.assertFalse(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    self.assertEqual(self._block.last_t.value, -1.0)
    self.assertEqual(self._block.last_fps.value, -1.0)

    Block.reset()
