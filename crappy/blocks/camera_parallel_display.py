# coding: utf-8

from multiprocessing.queues import Queue
from threading import Thread
from math import log2, ceil
import numpy as np
from typing import Optional
from time import time, sleep
import logging
import logging.handlers

from .camera_process import Camera_process
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


class Displayer(Camera_process):
  """"""

  def __init__(self,
               title: str,
               framerate: float,
               log_queue: Queue,
               log_level: int = 20,
               backend: Optional[str] = None,
               verbose: bool = False) -> None:
    """"""

    super().__init__(log_queue=log_queue,
                     log_level=log_level,
                     verbose=verbose)

    self._title = title
    self._framerate = framerate

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

  def _init(self) -> None:
    """"""

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

  def _get_data(self) -> bool:
    """"""

    with self._lock:

      if 'ImageUniqueID' not in self._data_dict:
        return False

      if self._data_dict['ImageUniqueID'] == self._metadata['ImageUniqueID'] \
          or time() - self._last_upd < 1 / self._framerate:
        return False

      self._metadata = self._data_dict.copy()
      self._last_upd = time()

      self._log(logging.DEBUG, f"Got new image to process with id "
                               f"{self._metadata['ImageUniqueID']}")

      np.copyto(self._img,
                np.frombuffer(self._img_array.get_obj(),
                              dtype=self._dtype).reshape(self._shape))

    return True

  def _loop(self) -> None:
    """"""

    if not self._get_data():
      return
    self.fps_count += 1

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

  def _finish(self) -> None:
    """"""

    self._log(logging.INFO, "Closing the displayer window")
    if self._backend == 'cv2':
      self._finish_cv2()
    elif self._backend == 'mpl':
      self._finish_mpl()

    if self._box_thread is not None and self._box_thread.is_alive():
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

    if self._title is not None:
      cv2.destroyWindow(self._title)

  def _finish_mpl(self) -> None:
    """Destroys the opened Matplotlib window."""

    if self._fig is not None:
      plt.close(self._fig)
