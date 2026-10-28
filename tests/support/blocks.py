"""Observable Block and Link doubles; fixture ownership lives separately."""

from collections.abc import Sequence
from multiprocessing import Event, Queue, Value
from typing import Any

from crappy import Block
from crappy.links.link import Link, ModifierType


class TestLink(Link):
  """Instrumented Link used in the Block tests.

  It behaves exactly like a regular Link, but also exposes Events allowing the
  tests to check which methods were actually called.
  """

  def __init__(self,
               input_block,
               output_block,
               modifiers: list[ModifierType] | None = None,
               name: str | None = None,
               allow_parallel: bool = False) -> None:
    """Initializes the tracking Events and the parent Link.

    Args:
      input_block: The upstream Block.
      output_block: The downstream Block.
      modifiers: Optional list of modifiers to apply to the transmitted data.
      name: Optional name for the Link.
      allow_parallel: Whether another Link may connect the same two Blocks.
    """

    self.polled = Event()
    self.sent = Event()
    self.received = Event()
    self.received_last = Event()
    self.received_chunk = Event()

    super().__init__(input_block, output_block, modifiers, name,
                     allow_parallel)

  def poll(self) -> bool:
    """Records that poll was called."""

    self.polled.set()
    return super().poll()

  def send(self, value: dict[str, Any]) -> None:
    """Records that send was called."""

    self.sent.set()
    return super().send(value)

  def recv(self) -> dict[str, Any]:
    """Records that recv was called."""

    self.received.set()
    return super().recv()

  def recv_last(self) -> dict[str, Any]:
    """Records that recv_last was called."""

    self.received_last.set()
    return super().recv_last()

  def recv_chunk(self) -> dict[str, list[Any]]:
    """Records that recv_chunk was called."""

    self.received_chunk.set()
    return super().recv_chunk()


def link(in_block,
         out_block,
         /, *,
         modifier: Sequence[ModifierType] | ModifierType | None = None,
         name: str | None = None,
         allow_parallel: bool = False) -> None:
  """Convenience wrapper creating a TestLink between two Blocks.

  Args:
    in_block: The upstream Block.
    out_block: The downstream Block.
    modifier: One modifier or a sequence of modifiers to attach to the Link.
    name: Optional name for the Link.
    allow_parallel: Whether another Link may connect the same two Blocks.
  """

  # Forcing the modifiers into lists so that the helper mirrors crappy.link.
  if modifier is not None:
    try:
      iter(modifier)
      modifier = list(modifier)
    except TypeError:
      modifier = [modifier]

  # Actually creating the Link object.
  TestLink(input_block=in_block,
           output_block=out_block,
           modifiers=modifier,
           name=name,
           allow_parallel=allow_parallel)


class TestBlock(Block):
  """Minimal Block used throughout the Block testing.

  It exposes Events and shared Values so that the tests can observe which
  lifecycle methods were called and what the internal timing attributes looked
  like at that moment.
  """

  test_instances = []
  test_queues = []

  def __init__(self, stop: bool = True, *, retain: bool = True) -> None:
    """Initializes the monitoring attributes.

    Args:
      stop: If :obj:`True`, the Block stops itself during the first loop.
      retain: Keep a fixture-owned reference until cleanup. Disable only when
        testing garbage collection or production ownership of a Block.
    """

    super().__init__()

    # These helpers exercise behavior, not terminal logging. Keeping the child
    # logger at CRITICAL avoids flooding test output with expected failures.
    self.debug = None

    self.prepared = Event()
    self.begun = Event()
    self.looped = Event()
    self.finished = Event()
    self.last_t = Value('d', -1.0)
    self.last_fps = Value('d', -1.0)
    self.n_loops = Value('i', 0)
    # Standalone run-cycle tests also need the flag normally set by prepare_all.
    self._main_entered = Value('b', 0, lock=False)
    self.test_failure_queue = Queue()
    self._failure_queue = self.test_failure_queue
    TestBlock.test_queues.append(self.test_failure_queue)
    if retain:
      TestBlock.test_instances.append(self)

    self.loops = Value('i', 0)

    self._stop = stop

  def prepare(self) -> None:
    """Records that prepare was reached and handles loop counting."""

    self.last_t.value = self._last_t
    self.last_fps.value = self._last_fps
    self.n_loops.value = self._n_loops

    self.prepared.set()

  def begin(self) -> None:
    """Records that begin was reached and handles loop counting."""

    self.last_t.value = self._last_t
    self.last_fps.value = self._last_fps
    self.n_loops.value = self._n_loops

    self.begun.set()

  def loop(self) -> None:
    """Records that loop was reached and optionally stops the Block."""

    self.last_t.value = self._last_t
    self.last_fps.value = self._last_fps
    self.n_loops.value = self._n_loops

    self.loops.value += 1

    self.looped.set()

    if self._stop:
      self.stop()

  def finish(self) -> None:
    """Records that finish was reached and handles loop counting."""

    self.last_t.value = self._last_t
    self.last_fps.value = self._last_fps
    self.n_loops.value = self._n_loops

    self.finished.set()
