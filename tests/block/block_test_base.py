# coding: utf-8

from crappy import Block
import unittest
from multiprocessing import Event, Value


class TestBlock(Block):
  """"""

  def __init__(self) -> None:
    """"""

    super().__init__()

    self.prepared = Event()
    self.begun = Event()
    self.looped = Event()
    self.finished = Event()
    self.last_t = Value('d', -1.0)
    self.last_fps = Value('d', -1.0)

  def prepare(self) -> None:
    """"""

    self.prepared.set()
    self.last_t.value = self._last_t if self._last_t is not None else -1.0
    self.last_fps.value = (self._last_fps
                           if self._last_fps is not None else -1.0)

  def begin(self) -> None:
    """"""

    self.begun.set()
    self.last_t.value = self._last_t if self._last_t is not None else -1.0
    self.last_fps.value = (self._last_fps
                           if self._last_fps is not None else -1.0)

  def loop(self) -> None:
    """"""

    self.looped.set()
    self.last_t.value = self._last_t if self._last_t is not None else -1.0
    self.last_fps.value = (self._last_fps
                           if self._last_fps is not None else -1.0)
    self.stop()

  def finish(self) -> None:
    """"""

    self.finished.set()
    self.last_t.value = self._last_t if self._last_t is not None else -1.0
    self.last_fps.value = (self._last_fps
                           if self._last_fps is not None else -1.0)


class BlockTestBase(unittest.TestCase):
  """"""

  def __init__(self, *args, **kwargs) -> None:
    """"""

    super().__init__(*args, **kwargs)

    self._block: TestBlock | None = None

  def tearDown(self) -> None:
    """"""

    if self._block is not None and self._block.is_alive():
      self._block.kill()
      self._block.join(3.0)
      if self._block.is_alive():
        self._block.terminate()
        raise RuntimeError("Block did not terminate as expected")

    self.assertEqual(0, len(Block.instances))
    self.assertEqual(0, len(Block.names))

    self.assertFalse(Block.thread_stop)
    self.assertFalse(Block.prepared_all)
    self.assertFalse(Block.launched_all)
    self.assertFalse(Block.no_raise)

    self.assertIsNone(Block.shared_t0)
    self.assertIsNone(Block.ready_barrier)
    self.assertIsNone(Block.start_event)
    self.assertIsNone(Block.pause_event)
    self.assertIsNone(Block.stop_event)
    self.assertIsNone(Block.raise_event)
    self.assertIsNone(Block.kbi_event)

    self.assertIsNone(Block.log_queue)
    self.assertIsNone(Block.log_thread)
