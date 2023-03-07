# coding: utf-8

from crappy import Block
import unittest
from multiprocessing import Value, Barrier, Event, Queue


class FakeException(Exception):
  """"""


class TestBlock(Block):
  """"""

  def __init__(self) -> None:
    """"""

    super().__init__()

    self.prepared = False
    self.begun = False
    self.looped = False
    self.finished = False

  def prepare(self) -> None:
    """"""

    self.prepared = True

  def begin(self) -> None:
    """"""

    self.begun = True

  def loop(self) -> None:
    """"""

    self.looped = True

  def finish(self) -> None:
    """"""

    self.finished = True


class BlockFailPrepare2(TestBlock):
  """"""

  def prepare(self) -> None:
    """"""

    super().prepare()

    raise FakeException


class BlockFailBegin2(TestBlock):
  """"""

  def begin(self) -> None:
    """"""

    super().begin()

    raise FakeException


class BlockFailLoop2(TestBlock):
  """"""

  def loop(self) -> None:
    """"""

    super().loop()

    raise FakeException


class BlockStop3(TestBlock):
  """"""

  def loop(self) -> None:
    """"""

    super().loop()
    self.stop()


class TestBlockFlow(unittest.TestCase):
  """"""

  def tearDown(self) -> None:
    """"""

    Block.reset()

  def test_flow_break_barrier(self) -> None:
    """"""

    block = BlockFailPrepare2()

    t0 = Value('d', 0)
    barrier = Barrier(1)
    start_event = Event()
    stop_event = Event()
    log_queue = Queue()

    block._instance_t0 = t0
    block._ready_barrier = barrier
    block._stop_event = stop_event
    block._start_event = start_event
    block._log_queue = log_queue
    block._log_level = None

    self.assertFalse(start_event.is_set())
    self.assertFalse(stop_event.is_set())
    self.assertFalse(barrier.broken)

    self.assertFalse(block.prepared)
    self.assertFalse(block.begun)
    self.assertFalse(block.looped)
    self.assertFalse(block.finished)

    block.run()

    self.assertTrue(block.prepared)
    self.assertFalse(block.begun)
    self.assertFalse(block.looped)
    self.assertTrue(block.finished)

    self.assertFalse(start_event.is_set())
    self.assertTrue(stop_event.is_set())
    self.assertTrue(barrier.broken)

  def test_flow_broken_barrier(self) -> None:
    """"""

    block = TestBlock()

    t0 = Value('d', 0)
    barrier = Barrier(1)
    start_event = Event()
    stop_event = Event()
    log_queue = Queue()

    block._instance_t0 = t0
    block._ready_barrier = barrier
    block._stop_event = stop_event
    block._start_event = start_event
    block._log_queue = log_queue
    block._log_level = None

    self.assertFalse(start_event.is_set())
    self.assertFalse(stop_event.is_set())
    self.assertFalse(barrier.broken)

    barrier.abort()
    self.assertTrue(barrier.broken)

    self.assertFalse(block.prepared)
    self.assertFalse(block.begun)
    self.assertFalse(block.looped)
    self.assertFalse(block.finished)

    block.run()

    self.assertTrue(block.prepared)
    self.assertFalse(block.begun)
    self.assertFalse(block.looped)
    self.assertTrue(block.finished)

    self.assertFalse(start_event.is_set())
    self.assertTrue(stop_event.is_set())
    self.assertTrue(barrier.broken)

  def test_flow_no_start(self) -> None:
    """"""

    block = TestBlock()

    t0 = Value('d', 0)
    barrier = Barrier(1)
    start_event = Event()
    stop_event = Event()
    log_queue = Queue()

    block._instance_t0 = t0
    block._ready_barrier = barrier
    block._stop_event = stop_event
    block._start_event = start_event
    block._log_queue = log_queue
    block._log_level = None

    self.assertFalse(start_event.is_set())
    self.assertFalse(stop_event.is_set())
    self.assertFalse(barrier.broken)

    self.assertFalse(block.prepared)
    self.assertFalse(block.begun)
    self.assertFalse(block.looped)
    self.assertFalse(block.finished)

    block.run()

    self.assertTrue(block.prepared)
    self.assertFalse(block.begun)
    self.assertFalse(block.looped)
    self.assertTrue(block.finished)

    self.assertFalse(start_event.is_set())
    self.assertTrue(stop_event.is_set())
    self.assertFalse(barrier.broken)

  def test_flow_fail_begin(self) -> None:
    """"""

    block = BlockFailBegin2()

    t0 = Value('d', 0)
    barrier = Barrier(1)
    start_event = Event()
    stop_event = Event()
    log_queue = Queue()

    block._instance_t0 = t0
    block._ready_barrier = barrier
    block._stop_event = stop_event
    block._start_event = start_event
    block._log_queue = log_queue
    block._log_level = None

    self.assertFalse(start_event.is_set())
    self.assertFalse(stop_event.is_set())
    self.assertFalse(barrier.broken)

    start_event.set()
    self.assertTrue(start_event.is_set())

    self.assertFalse(block.prepared)
    self.assertFalse(block.begun)
    self.assertFalse(block.looped)
    self.assertFalse(block.finished)

    block.run()

    self.assertTrue(block.prepared)
    self.assertTrue(block.begun)
    self.assertFalse(block.looped)
    self.assertTrue(block.finished)

    self.assertTrue(start_event.is_set())
    self.assertTrue(stop_event.is_set())
    self.assertFalse(barrier.broken)

  def test_flow_fail_loop(self) -> None:
    """"""

    block = BlockFailLoop2()

    t0 = Value('d', 0)
    barrier = Barrier(1)
    start_event = Event()
    stop_event = Event()
    log_queue = Queue()

    block._instance_t0 = t0
    block._ready_barrier = barrier
    block._stop_event = stop_event
    block._start_event = start_event
    block._log_queue = log_queue
    block._log_level = None

    self.assertFalse(start_event.is_set())
    self.assertFalse(stop_event.is_set())
    self.assertFalse(barrier.broken)

    start_event.set()
    self.assertTrue(start_event.is_set())

    self.assertFalse(block.prepared)
    self.assertFalse(block.begun)
    self.assertFalse(block.looped)
    self.assertFalse(block.finished)

    block.run()

    self.assertTrue(block.prepared)
    self.assertTrue(block.begun)
    self.assertTrue(block.looped)
    self.assertTrue(block.finished)

    self.assertTrue(start_event.is_set())
    self.assertTrue(stop_event.is_set())
    self.assertFalse(barrier.broken)

  def test_flow_stop(self) -> None:
    """"""

    block = BlockStop3()

    t0 = Value('d', 0)
    barrier = Barrier(1)
    start_event = Event()
    stop_event = Event()
    log_queue = Queue()

    block._instance_t0 = t0
    block._ready_barrier = barrier
    block._stop_event = stop_event
    block._start_event = start_event
    block._log_queue = log_queue
    block._log_level = None

    self.assertFalse(start_event.is_set())
    self.assertFalse(stop_event.is_set())
    self.assertFalse(barrier.broken)

    start_event.set()
    self.assertTrue(start_event.is_set())

    self.assertFalse(block.prepared)
    self.assertFalse(block.begun)
    self.assertFalse(block.looped)
    self.assertFalse(block.finished)

    block.run()

    self.assertTrue(block.prepared)
    self.assertTrue(block.begun)
    self.assertTrue(block.looped)
    self.assertTrue(block.finished)

    self.assertTrue(start_event.is_set())
    self.assertTrue(stop_event.is_set())
    self.assertFalse(barrier.broken)
