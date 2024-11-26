# coding: utf-8

import numpy as np
from multiprocessing import Process, current_process, get_start_method
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
import logging
import logging.handlers
from typing import Optional
from functools import partial
from time import sleep


class HistogramProcess(Process):
  """This class is a :obj:`multiprocessing.Process` taking an image as an input 
  via a :obj:`multiprocessing.Pipe`, and returning the histogram of that image 
  in another :obj:`~multiprocessing.Pipe`.

  It is used by the :class:`~crappy.tool.camera_config.CameraConfig` window and 
  its children to delegate and parallelize the calculation of the histogram. It
  allows to gain a few frames per second on the display in the configuration
  window.
  
  .. versionadded:: 2.0.0
  """

  def __init__(self,
               stop_event: Event,
               processing_event: Event,
               img_in: Queue,
               img_out: Queue,
               log_level: Optional[int],
               log_queue: Queue) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      stop_event: An :obj:`multiprocessing.Event` signaling the
        :obj:`~multiprocessing.Process` when to stop running.
      processing_event: An :obj:`multiprocessing.Event` set by the
        :obj:`multiprocessing.Process` to indicate that it's currently
        processing an image. Avoids having images to process piling up.
      img_in: The :class:`~multiprocessing.queues.Queue` through which
        the images to process are received.
      img_out: The :class:`~multiprocessing.queues.Queue` through which
        the calculated histograms are sent back.
      log_level: The minimum logging level of the entire Crappy script, as an
        :obj:`int`.
      log_queue: A :class:`multiprocessing.Queue` for sending the log messages
        to the main :obj:`~logging.Logger`, only used in Windows.
    """

    self._logger: Optional[logging.Logger] = None
    self._log_level = log_level
    self._log_queue = log_queue

    super().__init__(name=f"{current_process().name}.{type(self).__name__}")

    self._stop_event: Event = stop_event
    self._processing_event: Event = processing_event
    self._img_in: Queue = img_in
    self._img_out: Queue = img_out

  def run(self) -> None:
    """The main method being run by the HistogramProcess.

    It continuously receives images from the 
    :class:`~crappy.tool.camera_config.CameraConfig`, calculates their 
    histograms and returns them back as a nice image to integrate on the
    window.
    """

    try:
      self._processing_event.clear()

      # Initializing the variables
      img, auto_range, low_thresh, high_thresh = None, None, None, None

      # Looping until told to stop or an exception is raised
      while not self._stop_event.is_set():

        # Setting the processing event when busy processing an image
        if not self._img_in.empty():
          self._processing_event.set()
          # Receiving the image to process as well as additional parameters
          while not self._img_in.empty():
            (img, auto_range,
             low_thresh, high_thresh) = self._img_in.get_nowait()

          self.log(logging.DEBUG, "Received image from CameraConfig")

          # Calculating the histogram
          hist, _ = np.histogram(img, bins=np.arange(257))
          hist = np.repeat(hist / np.max(hist) * 80, 2)
          hist = np.repeat(hist[np.newaxis, :], 80, axis=0)

          # Making a nice image out of the calculated histogram
          out_img = np.fromfunction(partial(self._hist_func, histo=hist),
                                    shape=(80, 512))
          out_img = np.flip(out_img, axis=0).astype('uint8')

          # Adding vertical grey bars to indicate the limits of the auto range
          if auto_range:
            self.log(logging.DEBUG, "Drawing the line of the auto-range")
            out_img[:, round(2 * low_thresh)] = 127
            out_img[:, round(2 * high_thresh)] = 127

          # Sending back the histogram
          self._img_out.put_nowait(out_img)
          self._processing_event.clear()
          self.log(logging.DEBUG, "Sent the histogram back to the "
                                  "CameraConfig")

        # To avoid spamming the CPU in vain when idle
        else:
          sleep(0.001)

      self.log(logging.INFO, "Stop event set, stopping")

    except KeyboardInterrupt:
      self.log(logging.INFO, "Caught KeyboardInterrupt, stopping")
    except (Exception,) as exc:
      self._logger.exception("Caught Exception while running, stopping !",
                             exc_info=exc)
    finally:
      self.log(logging.INFO, "HistogramProcess finished")

  @staticmethod
  def _hist_func(x: np.ndarray,
                 _: np.ndarray,
                 histo: np.ndarray) -> np.ndarray:
    """Function passed to the :meth:`numpy.fromfunction` method for building 
    the histogram."""

    return np.where(x <= histo, 0, 255)

  def log(self, level: int, msg: str) -> None:
    """Records log messages for the HistogramProcess.

    Also instantiates the :obj:`~logging.Logger` when logging the first
    message.

    Args:
      level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.
    """

    if self._logger is None:
      self._set_logger()

    self._logger.log(level, msg)

  def _set_logger(self) -> None:
    """Instantiates and sets up the logger for the HistogramProcess."""

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
