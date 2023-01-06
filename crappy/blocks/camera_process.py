# coding: utf-8

from multiprocessing import Process, managers, get_start_method, \
  current_process
from multiprocessing.synchronize import Event, RLock, Barrier
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
from threading import BrokenBarrierError
import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Any
import logging
import logging.handlers
from select import select
from time import time

from ..links import Link
from .._global import LinkDataError
from ..tool import Spot_boxes


class Camera_process(Process):
  """"""

  def __init__(self,
               log_queue: Queue,
               log_level: int = 20) -> None:
    """"""

    super().__init__()
    self.name = f"{current_process().name}.{type(self).__name__}"

    self._log_queue = log_queue
    self._logger: Optional[logging.Logger] = None
    self._log_level = log_level

    self._img_array: Optional[SynchronizedArray] = None
    self._data_dict: Optional[managers.DictProxy] = None
    self._lock: Optional[RLock] = None
    self._cam_barrier: Optional[Barrier] = None
    self._stop_event: Optional[Event] = None
    self._shape: Optional[Tuple[int, int]] = None
    self._box_conn: Optional[Connection] = None
    self._outputs: List[Link] = list()
    self._labels: List[str] = list()

    self._img: Optional[np.ndarray] = None
    self._dtype = None
    self._metadata = {'ImageUniqueID': None}
    self._img0_set = False

    self._last_warn = time()

  def set_shared(self,
                 array: SynchronizedArray,
                 data_dict: managers.DictProxy,
                 lock: RLock,
                 barrier: Barrier,
                 event: Event,
                 shape: Tuple[int, int],
                 dtype,
                 box_conn: Optional[Connection],
                 outputs: List[Link],
                 labels: List[str]) -> None:
    """"""

    self._img_array = array
    self._data_dict = data_dict
    self._lock = lock
    self._cam_barrier = barrier
    self._stop_event = event
    self._shape = shape
    self._dtype = dtype
    self._box_conn = box_conn
    self._outputs = outputs
    self._labels = labels

    self._img = np.empty(shape=shape, dtype=dtype)

  def run(self) -> None:
    """"""

    try:
      self._set_logger()
      self._log(logging.INFO, "Logger configured")

      try:
        self._init()
      except (Exception,):
        self._cam_barrier.abort()
        self._log(logging.ERROR, "Breaking the barrier due to caught exception"
                                 " while preparing")
        raise

      self._log(logging.INFO, "Waiting for the other Camera processes to be "
                              "ready")
      self._cam_barrier.wait()
      self._log(logging.INFO, "All Camera processes ready now")

      while not self._stop_event.is_set():
        self._loop()

      self._log(logging.INFO, "Stop event set, stopping the processing")

    except KeyboardInterrupt:
      self._log(logging.INFO, "KeyboardInterrupt caught, stopping the "
                              "processing")

    except BrokenBarrierError:
      self._log(logging.WARNING,
                "Exception raised in another Camera process while waiting "
                "for all Camera processes to be ready, stopping")

    except (Exception,) as exc:
      self._logger.exception("Exception caught wile running !", exc_info=exc)
      self._log(logging.ERROR, "Setting the stop event to stop the other "
                               "Camera processes")
      self._stop_event.set()
      raise

    finally:
      self._finish()

  def _init(self) -> None:
    """"""

    ...

  def _get_data(self) -> bool:
    """"""

    with self._lock:

      if 'ImageUniqueID' not in self._data_dict:
        return False

      if self._data_dict['ImageUniqueID'] == self._metadata['ImageUniqueID']:
        return False

      self._metadata = self._data_dict.copy()

      self._log(logging.DEBUG, f"Got new image to process with id "
                               f"{self._metadata['ImageUniqueID']}")

      np.copyto(self._img,
                np.frombuffer(self._img_array.get_obj(),
                              dtype=self._dtype).reshape(self._shape))

    return True

  def _loop(self) -> None:
    """"""

    ...

  def _finish(self) -> None:
    """"""

    ...

  def _send(self, data: Union[list, Dict[str, Any]]) -> None:
    """"""

    # Building the dict to send from the data and labels if the data is a list
    if isinstance(data, list):
      if not self._labels:
        self._logger.log(logging.ERROR, "trying to send data as a list but no "
                                        "labels are specified ! Please add a "
                                        "self.labels attribute.")
        raise LinkDataError
      self._logger.log(logging.DEBUG, f"Converting {data} to dict before "
                                      f"sending")
      data = dict(zip(self._labels, data))

    # Making sure the data is being sent as a dict
    elif not isinstance(data, dict):
      self._logger.log(logging.ERROR, f"Trying to send a {type(data)} in a "
                                      f"Link !")
      raise LinkDataError

    # Sending the data to the downstream blocks
    for link in self._outputs:
      self._logger.log(logging.DEBUG, f"Sending {data} to Link {link}")
      link.send(data)

  def _send_box(self, boxes: Spot_boxes) -> None:
    """"""

    if self._box_conn is None:
      return

    self._log(logging.DEBUG, "Sending the box(es) to the displayer process")

    if select([], [self._box_conn], [], 0)[1]:
      self._box_conn.send(boxes)
    else:
      if time() - self._last_warn > 1:
        self._last_warn = time()
        self._log(logging.WARNING, f"Cannot send the box(es) to draw to the "
                                   f"Displayer process, the Pipe is full !")

  def _set_logger(self) -> None:
    """"""

    log_level = 10 * int(round(self._log_level / 10, 0))

    logger = logging.getLogger(self.name)
    logger.setLevel(min(log_level, logging.INFO))

    # On Windows, the messages need to be sent through a Queue for logging
    if get_start_method() == "spawn":
      queue_handler = logging.handlers.QueueHandler(self._log_queue)
      queue_handler.setLevel(min(log_level, logging.INFO))
      logger.addHandler(queue_handler)

    self._logger = logger

  def _log(self, level: int, msg: str) -> None:
    """Sends a log message to the logger.

    Args:
      level: The logging level, as an :obj:`int`.
      msg: The message to log, as a :obj:`str`.
    """

    if self._logger is None:
      return
    self._logger.log(level, msg)
