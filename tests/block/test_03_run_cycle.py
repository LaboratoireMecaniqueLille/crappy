# coding: utf-8

from crappy import Block
from multiprocessing import Barrier, Event, Value, Queue
from time import sleep

from .block_test_base import BlockTestBase, TestBlock


class TestBlockRaisePrepare(TestBlock):
  """"""

  def prepare(self) -> None:
    """"""

    super().prepare()
    raise ValueError


class TestBlockRaiseBegin(TestBlock):
  """"""

  def begin(self) -> None:
    """"""

    super().begin()
    raise ValueError


class TestBlockRaiseLoop(TestBlock):
  """"""

  def loop(self) -> None:
    """"""

    super().loop()
    raise ValueError


class TestBlockRaiseFinish(TestBlock):
  """"""

  def finish(self) -> None:
    """"""

    super().finish()
    raise ValueError


class TestRunCycle(BlockTestBase):
  """"""

  def test_normal_run(self) -> None:
    """"""

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

    self.assertGreater(self._block.last_t.value, -1.0)
    self.assertGreater(self._block.last_fps.value, -1.0)

    Block.reset()

  def test_raise_prepare(self) -> None:
    """"""

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

    sleep(0.5)

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
    """"""

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

    sleep(0.5)

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
    """"""

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

    sleep(0.5)

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
    """"""

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

    sleep(0.5)

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
