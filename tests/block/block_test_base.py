# coding: utf-8

from crappy import Block
from crappy.links.link import Link, ModifierType
from typing import Any
from collections.abc import Sequence
import logging
from time import monotonic, sleep
from typing import Callable
import unittest
from multiprocessing import Event, Value


class TestLink(Link):
  """Instrumented Link used in the Block tests.

  It behaves exactly like a regular Link, but also exposes Events allowing the
  tests to check which methods were actually called.
  """

  def __init__(self,
               input_block,
               output_block,
               modifiers: list[ModifierType] | None = None,
               name: str | None = None) -> None:
    """Initializes the tracking Events and the parent Link.

    Args:
      input_block: The upstream Block.
      output_block: The downstream Block.
      modifiers: Optional list of modifiers to apply to the transmitted data.
      name: Optional name for the Link.
    """

    self.polled = Event()
    self.sent = Event()
    self.received = Event()
    self.received_last = Event()
    self.received_chunk = Event()

    super().__init__(input_block, output_block, modifiers, name)

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
         name: str | None = None) -> None:
  """Convenience wrapper creating a TestLink between two Blocks.

  Args:
    in_block: The upstream Block.
    out_block: The downstream Block.
    modifier: One modifier or a sequence of modifiers to attach to the Link.
    name: Optional name for the Link.
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
           name=name)


class TestBlock(Block):
  """Minimal Block used throughout the Block testing.

  It exposes Events and shared Values so that the tests can observe which
  lifecycle methods were called and what the internal timing attributes looked
  like at that moment.
  """

  def __init__(self, stop: bool = True) -> None:
    """Initializes the monitoring attributes.

    Args:
      stop: If :obj:`True`, the Block stops itself during the first loop.
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


class BlockTestBase(unittest.TestCase):
  """Base test class shared by the Block unit tests.

  It mainly ensures that no stray Block process survives a test and that the
  class-level state of Block is fully reset after each run.
  """

  def __init__(self, *args, **kwargs) -> None:
    """Initializes the parent test case and the tracked Block reference."""

    super().__init__(*args, **kwargs)

    self._block: TestBlock | None = None

  def setUp(self) -> None:
    """Give every test an isolated, quiet Crappy logger."""

    logger = logging.getLogger('crappy')
    original_state = (list(logger.handlers), logger.level, logger.disabled,
                      logger.propagate, Block.logger,
                      Block.log_level,
                      logging.root.manager.disable)

    logger.handlers = list()
    logger.setLevel(logging.CRITICAL)
    logger.disabled = False
    logger.propagate = False
    Block.logger = None
    Block.log_level = logging.CRITICAL

    def restore_logging() -> None:
      (original_handlers, level, disabled, propagate, block_logger,
       block_log_level, disable) = original_state
      for handler in logger.handlers:
        if handler not in original_handlers:
          handler.close()
      logger.handlers = original_handlers
      logger.setLevel(level)
      logger.disabled = disabled
      logger.propagate = propagate
      Block.logger = block_logger
      Block.log_level = block_log_level
      logging.disable(disable)

    self.addCleanup(restore_logging)

  def wait_until(self,
                 predicate: Callable[[], bool],
                 timeout: float = 3.0) -> bool:
    """Poll a cross-process condition up to a deadline."""

    deadline = monotonic() + timeout
    while monotonic() < deadline:
      if predicate():
        return True
      sleep(0.005)
    return predicate()

  def wait_until_stable(self,
                        value_getter: Callable[[], int],
                        stable_for: float = 0.05,
                        timeout: float = 3.0) -> int:
    """Return once a shared counter has stopped changing for a short period."""

    deadline = monotonic() + timeout
    last_value = value_getter()
    unchanged_since = monotonic()

    while monotonic() < deadline:
      value = value_getter()
      now = monotonic()
      if value != last_value:
        last_value = value
        unchanged_since = now
      elif now - unchanged_since >= stable_for:
        return value
      sleep(0.005)

    self.fail("shared value did not stabilize before the deadline")

  def tearDown(self) -> None:
    """Kills the test Blocks if needed and checks that Block was reset."""

    # Make sure all Blocks are truly gone before leaving the test case.
    try:
      for inst in Block.instances:
        if inst.is_alive():
          inst.kill()
          inst.join(3.0)
          if inst.is_alive():
            inst.terminate()
            inst.join(1.0)

      self.assertFalse(any(inst.is_alive() for inst in Block.instances))

    finally:
      Block.reset()

    # Make sure the Block was properly reset
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
