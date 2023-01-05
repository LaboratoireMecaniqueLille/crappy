# coding: utf-8

from multiprocessing import Process, managers, get_start_method
from multiprocessing.synchronize import Event, RLock
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Any
from pathlib import Path
import logging
import logging.handlers
from ..tool import GPUCorrel
from ..links import Link
from .._global import LinkDataError


class Gpucorrel_parallel_process(Process):
  """"""

  def __init__(self,
               log_queue: Queue,
               parent_name: str,
               log_level: int = 20,
               discard_limit: float = 3,
               discard_ref: int = 5,
               calc_res: bool = False,
               img_ref: Optional[np.ndarray] = None,
               verbose: int = 0,
               levels: int = 5,
               resampling_factor: float = 2,
               kernel_file: Optional[Union[str, Path]] = None,
               iterations: int = 4,
               fields: Optional[List[str]] = None,
               mask: Optional[np.ndarray] = None,
               mul: float = 3) -> None:
    """"""

    super().__init__()

    self._gpucorrel_kw = dict(context=None,
                              verbose=verbose,
                              levels=levels,
                              resampling_factor=resampling_factor,
                              kernel_file=kernel_file,
                              iterations=iterations,
                              fields=fields,
                              ref_img=img_ref,
                              mask=mask,
                              mul=mul)

    self._log_queue = log_queue
    self._logger: Optional[logging.Logger] = None
    self._log_level = log_level
    self._parent_name = parent_name

    self._correl: Optional[GPUCorrel] = None
    self._img_ref = img_ref

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
    self._img0_set = img_ref is not None

    self._res_history = [np.inf]
    self._discard_limit = discard_limit
    self._discard_ref = discard_ref
    self._calc_res = calc_res

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

  def run(self) -> None:
    """"""

    try:
      self._set_logger()
      self._log(logging.INFO, "Logger configured")

      self._log(logging.INFO, "Instantiating the GPUCorrel tool")
      self._gpucorrel_kw.update(
        logger_name=f'crappy.{self._parent_name}.Process')
      self._correl = GPUCorrel(**self._gpucorrel_kw)

      if self._img_ref is not None:
        self._log(logging.INFO, "Initializing the GPUCorrel tool with the "
                                "given reference image")
        self._correl.set_img_size(self._img_ref.shape)
        self._correl.set_orig(self._img_ref.astype(np.float32))
        self._log(logging.INFO, "Preparing the GPUCorrel tool")
        self._correl.prepare()

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
            self._correl.set_img_size(self._img.shape)
            self._correl.set_orig(self._img.astype(np.float32))
            self._correl.prepare()
            self._img0_set = True
            continue

          self._log(logging.DEBUG, "Processing the received image")
          data = [self._metadata['t(s)'], self._metadata]
          data += self._correl.get_disp(self._img.astype(np.float32)).tolist()

          if self._calc_res:
            self._log(logging.DEBUG, "Calculating the residuals")
            res = self._correl.get_res()
            data.append(res)

            if self._discard_limit:
              self._log(logging.DEBUG, "Adding residuals to the residuals "
                                       "history")
              self._res_hist.append(res)
              self._res_hist = self._res_hist[-self._discard_ref - 1:]

              if res > self._discard_limit * np.average(self._res_hist[:-1]):
                self._log(logging.WARNING, "Residual too high, not sending "
                                           "values")
                continue

          self._send(data)

      self._log(logging.INFO, "Stop event set, stopping the processing")

    except KeyboardInterrupt:
      self._log(logging.INFO, "KeyboardInterrupt caught, stopping the "
                              "processing")

    finally:
      self._log(logging.INFO, "Cleaning up the GPUCorrel tool")
      self._correl.clean()

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
