# coding: utf-8

from multiprocessing import (Array, Barrier, Event, Manager, Pipe, Queue,
                             RLock, Value)
from multiprocessing.connection import Connection
from multiprocessing.managers import SyncManager
import multiprocessing.queues
from typing import Any, NamedTuple
import logging
import unittest

import numpy as np

from crappy._global import LinkDataError
from crappy.blocks.camera_processes.camera_process import CameraProcess


class SharedObjects(NamedTuple):
  """Multiprocessing objects shared with an instrumented CameraProcess."""

  array: Any
  data_dict: Any
  lock: Any
  barrier: Any
  stop_event: Any
  shape: tuple[int, int] | tuple[int, int, int]
  dtype: np.dtype
  to_draw_conn: Connection | None
  outputs: list[Any]
  labels: list[str] | None
  log_queue: Any


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
      raise ValueError

  def loop(self) -> None:
    """Records one loop iteration and optionally raises or stops."""

    self.looped.set()

    self.loops.value += 1
    if self.metadata['ImageUniqueID'] is not None:
      self.last_image_id.value = int(self.metadata['ImageUniqueID'])
    if self.img is not None:
      self.last_image_sum.value = float(np.sum(self.img))

    if self._raise_in == 'loop':
      raise ValueError

    if self._raise_in == 'keyboard':
      raise KeyboardInterrupt

    if self._stop_on_loop:
      self._stop_event.set()

  def finish(self) -> None:
    """Records that finish was reached."""

    self.finished.set()

    if self._raise_in == 'finish':
      raise ValueError


class CameraProcessTestBase(unittest.TestCase):
  """Base test class shared by CameraProcess unit tests."""

  def __init__(self, *args, **kwargs) -> None:
    """Initializes the parent test case and cleanup state."""

    super().__init__(*args, **kwargs)

    self._process: TestCameraProcess | None = None
    self._manager: SyncManager | None = None
    self._queues: list[multiprocessing.queues.Queue] = list()
    self._pipes: list[Connection] = list()

  def tearDown(self) -> None:
    """Stops child processes and releases multiprocessing resources."""

    try:
      if self._process is not None and self._process.is_alive():
        if self._process._stop_event is not None:
          self._process._stop_event.set()

        self._process.join(1.0)
        if self._process.is_alive():
          self._process.kill()
          self._process.join(1.0)

      if self._process is not None:
        self.assertFalse(self._process.is_alive())

    finally:
      for queue in self._queues:
        try:
          queue.cancel_join_thread()
          queue.close()
        except (AttributeError, OSError, ValueError):
          pass

      for pipe in self._pipes:
        try:
          pipe.close()
        except OSError:
          pass

      if self._manager is not None:
        self._manager.shutdown()

      logging.disable(logging.NOTSET)

  def make_shared(self,
                  process: TestCameraProcess | None = None,
                  shape: tuple[int, int] | tuple[int, int, int] = (3, 4),
                  dtype=np.uint8,
                  to_draw_conn: Connection | None = None,
                  outputs: list[TestLink] | None = None,
                  labels: list[str] | None = None,
                  log_level: int | None = logging.CRITICAL,
                  display_freq: bool = False,
                  barrier_parties: int = 1) -> SharedObjects:
    """Creates shared objects and passes them to a CameraProcess.

    Args:
      process: Process to configure. Defaults to ``self._process``.
      shape: Shape of the shared image.
      dtype: Dtype of the shared image.
      to_draw_conn: Optional Connection used for overlay messages.
      outputs: Optional Link-like outputs.
      labels: Optional labels for iterable data sent downstream.
      log_level: Logging level shared with the process.
      display_freq: Whether the process should log its processing frequency.
      barrier_parties: Number of parties in the camera-process Barrier.

    Returns:
      The shared objects passed to the process.
    """

    if process is None:
      if self._process is None:
        self._process = TestCameraProcess()
      process = self._process

    if self._manager is None:
      self._manager = Manager()

    dtype = np.dtype(dtype)
    array = Array(np.ctypeslib.as_ctypes_type(dtype), int(np.prod(shape)))
    data_dict = self._manager.dict()
    lock = RLock()
    barrier = Barrier(barrier_parties)
    stop_event = Event()
    log_queue = Queue()
    self._queues.append(log_queue)

    if outputs is None:
      outputs = list()

    process.set_shared(array=array,
                       data_dict=data_dict,
                       lock=lock,
                       barrier=barrier,
                       event=stop_event,
                       shape=shape,
                       dtype=dtype,
                       to_draw_conn=to_draw_conn,
                       outputs=outputs,
                       labels=labels,
                       log_queue=log_queue,
                       log_level=log_level,
                       display_freq=display_freq)

    return SharedObjects(array=array,
                         data_dict=data_dict,
                         lock=lock,
                         barrier=barrier,
                         stop_event=stop_event,
                         shape=shape,
                         dtype=dtype,
                         to_draw_conn=to_draw_conn,
                         outputs=outputs,
                         labels=labels,
                         log_queue=log_queue)

  def write_image(self,
                  shared: SharedObjects,
                  img: np.ndarray,
                  metadata: dict[str, Any] | None = None) -> None:
    """Writes a frame and its metadata into the shared objects."""

    if metadata is None:
      metadata = {'ImageUniqueID': 0, 't(s)': 0.0}

    np.copyto(np.frombuffer(shared.array.get_obj(),
                            dtype=shared.dtype).reshape(shared.shape),
              img)
    shared.data_dict.clear()
    shared.data_dict.update(metadata)

  def make_pipe(self) -> tuple[Connection, Connection]:
    """Creates a one-way Pipe and tracks both ends for cleanup."""

    recv_conn, send_conn = Pipe(duplex=False)
    self._pipes.extend((recv_conn, send_conn))
    return recv_conn, send_conn

  def set_test_logger(self, process: TestCameraProcess | None = None) -> None:
    """Attaches a quiet logger to a CameraProcess for direct method tests."""

    if process is None:
      process = self._process

    logger = logging.getLogger(process.name)
    logger.handlers.clear()
    logger.setLevel(logging.CRITICAL)
    process._logger = logger
