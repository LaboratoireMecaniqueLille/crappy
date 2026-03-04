# coding: utf-8

from crappy import Block
from crappy.links.link import Link, ModifierType
from typing import Any
from collections.abc import Sequence
import unittest
from multiprocessing import Event, Value


class TestLink(Link):
  """"""

  def __init__(self,
               input_block,
               output_block,
               modifiers: list[ModifierType] | None = None,
               name: str | None = None) -> None:
    """"""

    self.polled = Event()
    self.sent = Event()
    self.received = Event()
    self.received_last = Event()
    self.received_chunk = Event()

    super().__init__(input_block, output_block, modifiers, name)

  def poll(self) -> bool:
    """"""

    self.polled.set()
    return super().poll()

  def send(self, value: dict[str, Any]) -> None:
    """"""

    self.sent.set()
    return super().send(value)

  def recv(self) -> dict[str, Any]:
    """"""

    self.received.set()
    return super().recv()

  def recv_last(self) -> dict[str, Any]:
    """"""

    self.received_last.set()
    return super().recv_last()

  def recv_chunk(self) -> dict[str, list[Any]]:
    """"""

    self.received_chunk.set()
    return super().recv_chunk()


def link(in_block,
         out_block,
         /, *,
         modifier: Sequence[ModifierType] | ModifierType | None = None,
         name: str | None = None) -> None:
  """"""

  # Forcing the modifiers into lists
  if modifier is not None:
    try:
      iter(modifier)
      modifier = list(modifier)
    except TypeError:
      modifier = [modifier]

  # Actually creating the Link object
  TestLink(input_block=in_block,
           output_block=out_block,
           modifiers=modifier,
           name=name)


class TestBlock(Block):
  """"""

  def __init__(self, stop: bool = True) -> None:
    """"""

    super().__init__()

    self.prepared = Event()
    self.begun = Event()
    self.looped = Event()
    self.finished = Event()
    self.last_t = Value('d', -1.0)
    self.last_fps = Value('d', -1.0)
    self.n_loops = Value('i', 0)

    self.loops = Value('i', 0)

    self._stop = stop

  def prepare(self) -> None:
    """"""

    self.prepared.set()
    self.last_t.value = self._last_t if self._last_t is not None else -1.0
    self.last_fps.value = (self._last_fps
                           if self._last_fps is not None else -1.0)
    self.n_loops.value = self._n_loops

  def begin(self) -> None:
    """"""

    self.begun.set()
    self.last_t.value = self._last_t if self._last_t is not None else -1.0
    self.last_fps.value = (self._last_fps
                           if self._last_fps is not None else -1.0)
    self.n_loops.value = self._n_loops

  def loop(self) -> None:
    """"""

    self.looped.set()
    self.last_t.value = self._last_t if self._last_t is not None else -1.0
    self.last_fps.value = (self._last_fps
                           if self._last_fps is not None else -1.0)
    self.n_loops.value = self._n_loops

    self.loops.value += 1

    if self._stop:
      self.stop()

  def finish(self) -> None:
    """"""

    self.finished.set()
    self.last_t.value = self._last_t if self._last_t is not None else -1.0
    self.last_fps.value = (self._last_fps
                           if self._last_fps is not None else -1.0)
    self.n_loops.value = self._n_loops


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
