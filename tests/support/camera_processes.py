# coding: utf-8

from multiprocessing import Event, Value
from multiprocessing.connection import Connection
from typing import Any, NamedTuple

import numpy as np

from crappy._exceptions import LinkDataError
from crappy.blocks.camera_processes.camera_process import CameraProcess


class ExpectedProcessError(Exception):
  """Exception deliberately raised by the CameraProcess test fixture."""


class SharedObjects(NamedTuple):
  """Multiprocessing objects shared with an instrumented CameraProcess."""

  array: Any
  data_dict: Any
  lock: Any
  condition: Any
  barrier: Any
  stop_event: Any
  shape: tuple[int, int] | tuple[int, int, int]
  dtype: np.dtype
  to_draw_conn: Connection | None
  outputs: list[Any]
  labels: list[str] | None
  log_queue: Any
  failure_queue: Any


class TestLink:
  """Minimal Link-like object recording values sent by CameraProcess."""

  def __init__(self, name: str = "test_link") -> None:
    """Initializes the Event and in-memory record of sent values."""

    self.name = name
    self.sent = Event()
    self.sent_values: list[dict[str, Any]] = list()

  def send(self, value: dict[str, Any]) -> None:
    """Records sent values and mirrors Link's dict-only contract."""

    if not isinstance(value, dict):
      raise LinkDataError

    self.sent.set()
    self.sent_values.append(value.copy())


class TestCameraProcess(CameraProcess):
  """Instrumented CameraProcess used throughout the CameraProcess tests."""

  def __init__(self,
               stop_on_loop: bool = True,
               raise_in: str | None = None) -> None:
    """Initializes monitoring Events and shared counters.

    Args:
      stop_on_loop: If :obj:`True`, the process stops itself on first loop.
      raise_in: Optional lifecycle method in which an exception is raised.
    """

    super().__init__()

    self.initialized = Event()
    self.looped = Event()
    self.finished = Event()

    self.loops = Value('i', 0)
    self.last_image_id = Value('i', -1)
    self.last_image_sum = Value('d', -1.0)

    self._stop_on_loop = stop_on_loop
    self._raise_in = raise_in

  def init(self) -> None:
    """Records initialization and optionally raises."""

    self.initialized.set()

    if self._raise_in == 'init':
      raise ExpectedProcessError

  def loop(self) -> None:
    """Records one loop iteration and optionally raises or stops."""

    self.looped.set()

    self.loops.value += 1
    if self.metadata['ImageUniqueID'] is not None:
      self.last_image_id.value = int(self.metadata['ImageUniqueID'])
    if self.img is not None:
      self.last_image_sum.value = float(np.sum(self.img))

    if self._raise_in == 'loop':
      raise ExpectedProcessError

    if self._raise_in == 'keyboard':
      raise KeyboardInterrupt

    if self._stop_on_loop:
      self._stop_event.set()

  def finish(self) -> None:
    """Records that finish was reached."""

    self.finished.set()

    if self._raise_in == 'finish':
      raise ValueError
