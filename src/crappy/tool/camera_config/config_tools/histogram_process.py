# coding: utf-8

"""This file contains the code for the process calculating the histogram on the
camera configuration window."""

import numpy as np
from multiprocessing import Process, current_process, get_start_method
from multiprocessing.synchronize import Event
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
import logging
import logging.handlers
from typing import Optional
from functools import partial
from time import sleep


class HistogramProcess(Process):
  """This class is a process taking an image as an input, and returning the
  histogram of that image."""

  def __init__(self,
               stop_event: Event,
               processing_event: Event,
               img_in: Connection,
               img_out: Connection,
               log_level: Optional[int],
               log_queue: Queue) -> None:
    """"""

    self._logger: Optional[logging.Logger] = None
    self._log_level = log_level
    self._log_queue = log_queue

    super().__init__(name=f"{current_process().name}.{type(self).__name__}")

    self._stop_event: Event = stop_event
    self._processing_event: Event = processing_event
    self._img_in: Connection = img_in
    self._img_out: Connection = img_out

  def run(self) -> None:
    """"""

    try:
      self._processing_event.clear()

      while not self._stop_event.is_set():

        if self._img_in.poll():

          self._processing_event.set()
          while self._img_in.poll():
            img, auto_range, low_thresh, high_thresh = self._img_in.recv()

          self.log(logging.DEBUG, "Received image from CameraConfig")

          hist, _ = np.histogram(img, bins=np.arange(257))
          hist = np.repeat(hist / np.max(hist) * 80, 2)
          hist = np.repeat(hist[np.newaxis, :], 80, axis=0)

          out_img = np.fromfunction(partial(self._hist_func, histo=hist),
                                    shape=(80, 512))
          out_img = np.flip(out_img, axis=0).astype('uint8')

          # Adding vertical grey bars to indicate the limits of the auto range
          if auto_range:
            self.log(logging.DEBUG, "Drawing the line of the auto-range")
            out_img[:, round(2 * low_thresh)] = 127
            out_img[:, round(2 * high_thresh)] = 127

          self._img_out.send(out_img)
          self._processing_event.clear()
          self.log(logging.DEBUG, "Sent the histogram back to the "
                                  "CameraConfig")

      else:
        sleep(0.001)

      self.log(logging.INFO, "Stop event set, stopping")

    except KeyboardInterrupt:
      self.log(logging.INFO, "Caught KeyboardInterrupt, stopping")

  @staticmethod
  def _hist_func(x: np.ndarray,
                 _: np.ndarray,
                 histo: np.ndarray) -> np.ndarray:
    """Function passed to the :meth:`np.fromfunction` method for building the
    histogram."""

    return np.where(x <= histo, 0, 255)

  def log(self, level: int, msg: str) -> None:
    """"""

    if self._logger is None:
      self._set_logger()

    self._logger.log(level, msg)

  def _set_logger(self) -> None:
    """"""

    logger = logging.getLogger(self.name)

    # Disabling logging if requested
    if self._log_level is not None:
      logger.setLevel(self._log_level)
    else:
      logging.disable()

    # On Windows, the messages need to be sent through a Queue for logging
    if get_start_method() == "spawn" and self._log_level is not None:
      queue_handler = logging.handlers.QueueHandler(self._log_queue)
      queue_handler.setLevel(self._log_level)
      logger.addHandler(queue_handler)

    self._logger = logger
