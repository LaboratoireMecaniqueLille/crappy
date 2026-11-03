"""Ownership of shared objects used by CameraProcess tests."""

from multiprocessing import Array, Barrier, Condition, Event, Manager, Pipe, Queue, RLock, current_process
import logging
import unittest
from unittest.mock import Mock
import numpy as np

from crappy._failure import FailureManager
from tests.support.camera_processes import SharedObjects, TestCameraProcess, TestLink
from tests.support.processes import stop_process
from tests.support.resources import close_queue, drain_queue, isolate_logger, run_cleanups
from multiprocessing.managers import SyncManager
from multiprocessing.connection import Connection
import multiprocessing.queues
from typing import Any


class CameraProcessTestBase(unittest.TestCase):
  """Base test class shared by CameraProcess unit tests."""

  def __init__(self, *args, **kwargs) -> None:
    """Initializes the parent test case and cleanup state."""

    super().__init__(*args, **kwargs)

    self._process: TestCameraProcess | None = None
    self._manager: SyncManager | None = None
    self._queues: list[multiprocessing.queues.Queue] = list()
    self._pipes: list[Connection] = list()

  def setUp(self):
    super().setUp()
    isolate_logger(self, f"{current_process().name}.TestCameraProcess")
    self._processes = []
    self.addCleanup(self._cleanup_resources)

  def _cleanup_resources(self):
    processes = list(dict.fromkeys([*self._processes,
                                    *([self._process] if self._process is not None else [])]))

    def stop(process):
      def request():
        if process._stop_event is not None:
          process._stop_event.set()
        if process._condition is not None:
          with process._condition:
            process._condition.notify_all()
      stop_process(process, request,
                   drain=lambda: [drain_queue(queue) for queue in self._queues])

    run_cleanups(*(lambda process=process: stop(process) for process in processes),
                 *(pipe.close for pipe in self._pipes),
                 *(lambda queue=queue: close_queue(queue) for queue in self._queues),
                 *([self._manager.shutdown] if self._manager is not None else []))

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

    if process not in self._processes:
      self._processes.append(process)

    if self._manager is None:
      self._manager = Manager()

    dtype = np.dtype(dtype)
    array = Array(np.ctypeslib.as_ctypes_type(dtype), int(np.prod(shape)))
    data_dict = self._manager.dict()
    lock = RLock()
    condition = Condition(lock)
    barrier = Barrier(barrier_parties)
    stop_event = Event()
    log_queue = Queue()
    self._queues.append(log_queue)
    failure_queue = Queue()
    self._queues.append(failure_queue)

    if outputs is None:
      outputs = list()

    process.set_shared(array=array,
                       data_dict=data_dict,
                       lock=lock,
                       condition=condition,
                       barrier=barrier,
                       event=stop_event,
                       shape=shape,
                       dtype=dtype,
                       to_draw_conn=to_draw_conn,
                       outputs=outputs,
                       labels=labels,
                       log_queue=log_queue,
                       failure_queue=failure_queue,
                       block_name='test Camera',
                       log_level=log_level,
                       display_freq=display_freq)

    return SharedObjects(array=array,
                         data_dict=data_dict,
                         lock=lock,
                         condition=condition,
                         barrier=barrier,
                         stop_event=stop_event,
                         shape=shape,
                         dtype=dtype,
                         to_draw_conn=to_draw_conn,
                         outputs=outputs,
                         labels=labels,
                         log_queue=log_queue,
                         failure_queue=failure_queue)

  def receive_failures(self, shared: SharedObjects) -> FailureManager:
    """Drains flushed child reports without logging their receipt."""

    manager = FailureManager('test owner', 'main')
    manager._logger = Mock()
    while manager.receive((shared.failure_queue,)):
      pass
    return manager

  def write_image(self,
                  shared: SharedObjects,
                  img: np.ndarray,
                  metadata: dict[str, Any] | None = None) -> None:
    """Writes a frame and its metadata into the shared objects."""

    if metadata is None:
      metadata = {'ImageUniqueID': 0, 't(s)': 0.0}

    with shared.condition:
      np.copyto(np.frombuffer(shared.array.get_obj(),
                              dtype=shared.dtype).reshape(shared.shape),
                img)
      shared.data_dict.clear()
      shared.data_dict.update(metadata)
      shared.condition.notify_all()

  def make_pipe(self) -> tuple[Connection, Connection]:
    """Creates a one-way Pipe and tracks both ends for cleanup."""

    recv_conn, send_conn = Pipe(duplex=False)
    self._pipes.extend((recv_conn, send_conn))
    return recv_conn, send_conn

  def set_test_logger(self, process: TestCameraProcess | None = None) -> None:
    """Attaches a quiet logger to a CameraProcess for direct method tests."""

    if process is None:
      process = self._process

    logger = isolate_logger(self, process.name, logging.CRITICAL)
    process._logger = logger
    process._failure_manager = FailureManager('test Camera', 'main',
                                              owner_path=process.name)
    process._failure_manager._logger = logger
