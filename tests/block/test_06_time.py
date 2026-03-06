# coding: utf-8

from crappy import Block
from crappy._global import T0NotSetError
from multiprocessing import Barrier, Event, Value, Queue
from time import sleep

from .block_test_base import BlockTestBase, TestBlock


class TestBlockT0(TestBlock):
  """Test Block accessing t0 from inside loop."""

  def loop(self) -> None:
    """Runs the regular loop body and then reads t0."""

    super().loop()
    _ = self.t0


class TestBlockTime(BlockTestBase):
  """Tests the handling of the shared start time in Blocks."""

  def test_missing_t0(self) -> None:
    """Tests that reading t0 before it is set raises an error."""

    self._block = TestBlockT0(stop=False)

    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = Event()
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = Event()
    self._block._instance_t0 = Value('d', -1.0)
    self._block._log_queue = Queue()

    self._block._start_event.set()

    with self.assertRaises(T0NotSetError):
      _ = self._block.t0

    self._block.start()

    self._block.join(4.0)

    # The error should be converted into the shared raise Event by Block.run
    self.assertTrue(self._block._start_event.is_set())
    self.assertTrue(self._block._stop_event.is_set())
    self.assertFalse(self._block._ready_barrier.broken)
    self.assertTrue(self._block._raise_event.is_set())
    self.assertFalse(self._block._kbi_event.is_set())

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    Block.reset()

  def test_start_event(self) -> None:
    """Tests the timeout path when the start Event is never set."""

    self._block = TestBlockT0(stop=False)

    self._block._ready_barrier = Barrier(1)
    self._block._start_event = Event()
    self._block._stop_event = Event()
    self._block._raise_event = Event()
    self._block._kbi_event = Event()
    self._block._pause_event = Event()
    self._block._instance_t0 = Value('d', -1.0)
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

    Block.reset()

  def test_t0_normal(self) -> None:
    """Tests that t0 is available during a normal Block launch."""

    self._block = TestBlock()

    Block.prepare_all()

    sleep(0.5)

    Block.launch_all()

    sleep(0.5)

    self.assertTrue(self._block._start_event.is_set())
    self.assertTrue(self._block._stop_event.is_set())
    self.assertFalse(self._block._ready_barrier.broken)
    self.assertFalse(self._block._raise_event.is_set())
    self.assertFalse(self._block._kbi_event.is_set())

    self.assertTrue(self._block.prepared.is_set())
    self.assertTrue(self._block.begun.is_set())
    self.assertTrue(self._block.looped.is_set())
    self.assertTrue(self._block.finished.is_set())

    self.assertGreater(self._block.t0, 0)

    Block.reset()
