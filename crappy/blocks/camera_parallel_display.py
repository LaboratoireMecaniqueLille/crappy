# coding: utf-8

from multiprocessing import Process, managers
from multiprocessing.synchronize import Event, RLock
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.connection import Connection
from multiprocessing.queues import Queue
from threading import Thread
from math import log2, ceil
import numpy as np
from typing import Optional, Tuple
from time import time, sleep
import logging
import logging.handlers
from .._global import OptionalModule
from ..tool import Spot_boxes, Box

try:
  from PIL import ImageTk, Image
except (ModuleNotFoundError, ImportError):
  ImageTk = OptionalModule("Pillow")
  Image = OptionalModule("Pillow")

try:
  import matplotlib.pyplot as plt
except (ModuleNotFoundError, ImportError):
  plt = OptionalModule("matplotlib")

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class Displayer(Process):
  """"""

  def __init__(self,
               title: str,
               framerate: float,
               log_queue: Queue,
               parent_name: str,
               log_level: int = 20,
               backend: Optional[str] = None) -> None:
    """"""

    super().__init__()

    self._title = title
    self._framerate = framerate

    self._log_queue = log_queue
    self._logger: Optional[logging.Logger] = None
    self._log_level = log_level
    self._parent_name = parent_name

    # Selecting the backend if no backend was specified
    if backend is None:
      if not isinstance(cv2, OptionalModule):
        self._backend = 'cv2'
      elif not isinstance(plt, OptionalModule):
        self._backend = 'mpl'
      else:
        raise ModuleNotFoundError("Neither opencv-python nor matplotlib could "
                                  "be imported, no backend found for "
                                  "displaying the images")

    elif backend in ('cv2', 'mpl'):
      self._backend = backend
    else:
      raise ValueError("The backend argument should be either 'cv2' or "
                       "'mpl' !")

    # Setting other instance attributes
    self._ax = None
    self._fig = None
    self._last_upd = time()

    self._img_array: Optional[SynchronizedArray] = None
    self._data_dict: Optional[managers.DictProxy] = None
    self._lock: Optional[RLock] = None
    self._stop_event: Optional[Event] = None
    self._shape: Optional[Tuple[int, int]] = None
    self._box_conn: Optional[Connection] = None

    self._img: Optional[np.ndarray] = None
    self._last_nr = None
    self._dtype = None

    # The thread must be initialized later for compatibility with Windows
    self._box_thread: Optional[Thread] = None

    self._boxes: Spot_boxes = Spot_boxes()
    self._stop_thread = False

  def __del__(self) -> None:
    """"""

    if self._box_thread is not None and self._box_thread.is_alive():
      self._stop_thread = True
      try:
        self._box_thread.join(0.05)
      except RuntimeError:
        pass

  def set_shared(self,
                 array: SynchronizedArray,
                 data_dict: managers.DictProxy,
                 lock: RLock,
                 event: Event,
                 shape: Tuple[int, int],
                 dtype,
                 box_conn: Connection) -> None:
    """"""

    self._img_array = array
    self._data_dict = data_dict
    self._lock = lock
    self._stop_event = event
    self._shape = shape
    self._dtype = dtype
    self._box_conn = box_conn

    self._img = np.empty(shape=shape, dtype=dtype)

  def run(self) -> None:
    """"""

    try:
      self._set_logger()
      self._log(logging.INFO, "Logger configured")

      self._log(logging.INFO, "Instantiating the thread for getting the boxes "
                              "to display")
      self._box_thread = Thread(target=self._thread_target)
      self._log(logging.INFO, "Starting the thread for getting the boxes to "
                              "display")
      self._box_thread.start()

      self._log(logging.INFO, f"Opening the displayer window with the backend "
                              f"{self._backend}")
      if self._backend == 'cv2':
        self._prepare_cv2()
      elif self._backend == 'mpl':
        self._prepare_mpl()

      while not self._stop_event.is_set():
        display = False
        with self._lock:

          if 'ImageUniqueID' not in self._data_dict:
            continue

          if self._data_dict['ImageUniqueID'] != self._last_nr and \
              time() - self._last_upd >= 1 / self._framerate:
            self._last_nr = self._data_dict['ImageUniqueID']
            display = True

            self._log(logging.DEBUG, f"Got new image to display with id "
                                     f"{self._last_nr}")

            np.copyto(self._img,
                      np.frombuffer(self._img_array.get_obj(),
                                    dtype=self._dtype).reshape(self._shape))

        if display:

          # Casts the image to uint8 if it's not already in this format
          if self._img.dtype != np.uint8:
            self._log(logging.DEBUG, f"Casting displayed image from "
                                     f"{self._img.dtype} to uint8")
            if np.max(self._img) > 255:
              factor = max(ceil(log2(np.max(self._img) + 1) - 8), 0)
              img = (self._img / 2 ** factor).astype(np.uint8)
            else:
              img = self._img.astype(np.uint8)
          else:
            img = self._img.copy()

          # Drawing the latest known position of the boxes
          for box in self._boxes:
            if box is not None:
              self._log(logging.DEBUG, "Drawing boxes on top of the image to "
                                       "display")
              self._draw_box(img, box)

          # Calling the right prepare method
          if self._backend == 'cv2':
            self._update_cv2(img)
          elif self._backend == 'mpl':
            self._update_mpl(img)

      self._log(logging.INFO, "Stop event set, stopping the display")

    except KeyboardInterrupt:
      self._log(logging.INFO, "KeyboardInterrupt caught, stopping the display")

    finally:
      if self._backend == 'cv2':
        self._finish_cv2()
      elif self._backend == 'mpl':
        self._finish_mpl()

      if self._box_thread.is_alive():
        self._stop_thread = True
        try:
          self._box_thread.join(0.05)
        except RuntimeError:
          self._log(logging.WARNING, "Thread for receiving the boxes did not "
                                     "stop as expected")

  def _thread_target(self) -> None:
    """"""

    while not self._stop_event.is_set() and not self._stop_thread:

      boxes = None
      while self._box_conn.poll():
        boxes = self._box_conn.recv()

      if boxes is not None:
        self._log(logging.DEBUG, f"Received boxes to display: {boxes}")
        self._boxes = boxes

      else:
        sleep(0.001)

    self._log(logging.INFO, "Thread for receiving the boxes ended")

  def _set_logger(self) -> None:
    """"""

    log_level = 10 * int(round(self._log_level / 10, 0))

    logger = logging.getLogger(f'crappy.{self._parent_name}.Displayer')
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

  def _draw_box(self, img: np.ndarray, box: Box) -> None:
    """Draws a box on top of an image."""

    if box.no_points():
      return

    x_top, x_bottom, y_left, y_right = box.sorted()

    try:
      for line in ((box.y_start, slice(x_top, x_bottom)),
                   (box.y_end, slice(x_top, x_bottom)),
                   (slice(y_left, y_right), x_top),
                   (slice(y_left, y_right), x_bottom),
                   (box.y_start + 1, slice(x_top, x_bottom)),
                   (box.y_end - 1, slice(x_top, x_bottom)),
                   (slice(y_left, y_right), x_top + 1),
                   (slice(y_left, y_right), x_bottom - 1)
                   ):
        img[line] = 255 * int(np.mean(img[line]) < 128)
    except (Exception,) as exc:
      self._logger.exception("Encountered exception while drawing boxes, "
                             "ignoring", exc_info=exc)

  def _prepare_cv2(self) -> None:
    """Instantiates the display window of cv2."""

    try:
      flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    except AttributeError:
      flags = cv2.WINDOW_NORMAL
    cv2.namedWindow(self._title, flags)

  def _prepare_mpl(self) -> None:
    """Creates a Matplotlib figure."""

    plt.ion()
    self._fig, self._ax = plt.subplots()

  def _update_cv2(self, img: np.ndarray) -> None:
    """Reshapes the image to a maximum shape of 640x480 and displays it."""

    if img.shape[0] > 480 or img.shape[1] > 640:
      factor = min(480 / img.shape[0], 640 / img.shape[1])
      self._log(
        logging.DEBUG,
        f"Reshaping displayed image from {img.shape} to "
        f"{int(img.shape[1] * factor), int(img.shape[0] * factor)}")
      img = cv2.resize(img, (int(img.shape[1] * factor),
                             int(img.shape[0] * factor)))

    self._log(logging.DEBUG, "Displaying the image")
    cv2.imshow(self._title, img)
    cv2.waitKey(1)

  def _update_mpl(self, img: np.ndarray) -> None:
    """Reshapes the image to a dimension inferior or equal to 640x480 and
    displays it."""

    if img.shape[0] > 480 or img.shape[1] > 640:
      factor = max(ceil(img.shape[0] / 480), ceil(img.shape[1] / 640))
      self._log(
        logging.DEBUG,
        f"Reshaping the displayed image from {img.shape} to "
        f"{(img.shape[0] / factor, img.shape[1] / factor)}")
      img = img[::factor, ::factor]

    self._ax.clear()
    self._log(logging.DEBUG, "Displaying the image")
    self._ax.imshow(img, cmap='gray')
    plt.pause(0.001)
    plt.show()

  def _finish_cv2(self) -> None:
    """Destroys the opened cv2 window."""

    cv2.destroyWindow(self._title)

  def _finish_mpl(self) -> None:
    """Destroys the opened Matplotlib window."""

    plt.close(self._fig)
