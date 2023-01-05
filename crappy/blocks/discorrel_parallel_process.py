# coding: utf-8

from multiprocessing import Process, managers, get_start_method
from multiprocessing.synchronize import Event, RLock
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Any
import logging
import logging.handlers
from ..tool import DISCorrel, Box, Spot_boxes
from ..links import Link
from .._global import LinkDataError


class Discorrel_parallel_process(Process):
  """"""

  def __init__(self,
               log_queue: Queue,
               parent_name: str,
               log_level: int = 20,
               fields: List[str] = None,
               alpha: float = 3,
               delta: float = 1,
               gamma: float = 0,
               finest_scale: int = 1,
               iterations: int = 1,
               gradient_iterations: int = 10,
               init: bool = True,
               patch_size: int = 8,
               patch_stride: int = 3,
               residual: bool = False,) -> None:
    """"""

    super().__init__()

    self._log_queue = log_queue
    self._logger: Optional[logging.Logger] = None
    self._log_level = log_level
    self._parent_name = parent_name

    self._discorrel_kw = dict(fields=fields,
                              alpha=alpha,
                              delta=delta,
                              gamma=gamma,
                              finest_scale=finest_scale,
                              init=init,
                              iterations=iterations,
                              gradient_iterations=gradient_iterations,
                              patch_size=patch_size,
                              patch_stride=patch_stride)
    self._residual = residual

    self._img_array: Optional[SynchronizedArray] = None
    self._data_dict: Optional[managers.DictProxy] = None
    self._lock: Optional[RLock] = None
    self._stop_event: Optional[Event] = None
    self._shape: Optional[Tuple[int, int]] = None
    self._box_conn: Optional[Connection] = None
    self._outputs: List[Link] = list()
    self._labels: List[str] = list()

    self._img: Optional[np.ndarray] = None
    self._dtype = None
    self._metadata = {'ImageUniqueID': None}
    self._img0_set = False

  def set_shared(self,
                 array: SynchronizedArray,
                 data_dict: managers.DictProxy,
                 lock: RLock,
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
    self._stop_event = event
    self._shape = shape
    self._dtype = dtype
    self._box_conn = box_conn
    self._outputs = outputs
    self._labels = labels

    self._img = np.empty(shape=shape, dtype=dtype)

  def set_box(self, box: Box) -> None:
    """"""

    self._discorrel_kw.update(dict(box=box))

  def run(self) -> None:
    """"""

    try:
      self._set_logger()
      self._log(logging.INFO, "Logger configured")

      self._log(logging.INFO, "Instantiating the Discorrel tool")
      self._discorrel = DISCorrel(**self._discorrel_kw)

      while not self._stop_event.is_set():
        process = False
        with self._lock:

          if 'ImageUniqueID' not in self._data_dict:
            continue

          if self._data_dict['ImageUniqueID'] != \
              self._metadata['ImageUniqueID']:
            self._metadata = self._data_dict.copy()
            process = True

            self._log(logging.DEBUG, f"Got new image to process with id "
                                     f"{self._metadata['ImageUniqueID']}")

            np.copyto(self._img,
                      np.frombuffer(self._img_array.get_obj(),
                                    dtype=self._dtype).reshape(self._shape))

        if process:

          if not self._img0_set:
            self._log(logging.INFO, "Setting the reference image")
            self._discorrel.set_img0(np.copy(self._img))
            self._img0_set = True
            continue

          self._log(logging.DEBUG, "Processing the received image")
          data = self._discorrel.get_data(self._img, self._residual)
          self._send([self._metadata['t(s)'], self._metadata, *data])

          if self._box_conn is not None:
            self._log(logging.DEBUG, "Sending the box to the displayer "
                                     "process")
            self._box_conn.send(Spot_boxes(self._discorrel.box))

      self._log(logging.INFO, "Stop event set, stopping the processing")

    except KeyboardInterrupt:
      self._log(logging.INFO, "KeyboardInterrupt caught, stopping the "
                              "processing")

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

  def _set_logger(self) -> None:
    """"""

    log_level = 10 * int(round(self._log_level / 10, 0))

    logger = logging.getLogger(f'crappy.{self._parent_name}.Process')
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
