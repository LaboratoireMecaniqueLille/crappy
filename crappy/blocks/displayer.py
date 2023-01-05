# coding: utf-8

import numpy as np
from typing import Optional
from time import time
import logging
from multiprocessing import current_process
from .._global import OptionalModule

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


class Displayer:
  """This class manages the display of images captured by a camera object.

  It can either work with Matplotlib or OpenCV as a backend, OpenCV being the
  fastest. It tries to display images at a given framerate, dropping part of
  the received frames if necessary.

  The images are displayed with a maximum resolution of 640x480, and are
  resized to match that resolution if necessary. Similarly, the maximum bit
  depth is 8 bits, and the images are cast if necessary. Resizing and casting
  are anyway less demanding on the CPU than displaying big images.
  """

  def __init__(self,
               title: str,
               framerate: float,
               backend: Optional[str] = None) -> None:
    """Sets the args and the other instance attributes, and looks for an
    available display backend if none was specified.

    Args:
      title: The title to display on the display window.
      framerate: The maximum framerate for displaying the images. To achieve
        this framerate, part of the received frames are simply dropped.
      backend: The backend to use for displaying the images. Should be one of:
        ::

          'cv2', 'mpl'
    """

    self._title = title
    self._framerate = framerate
    self._logger: Optional[logging.Logger] = None

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

  def prepare(self) -> None:
    """Calls the right prepare method depending on the chosen backend."""

    self._log(logging.INFO, f"Opening the displayer window with the backend "
                            f"{self._backend}")
    if self._backend == 'cv2':
      self._prepare_cv2()
    elif self._backend == 'mpl':
      self._prepare_mpl()

  def update(self, img: np.ndarray) -> None:
    """Ensures the target framerate is respected, and calls the right update
    method depending on the chosen backend.

    Args:
      img: The image to display on the displayer.
    """

    # Making sure the image is not being refreshed too often
    if time() - self._last_upd < 1 / self._framerate:
      return
    else:
      self._last_upd = time()

    # Casts the image to uint8 if it's not already in this format
    if img.dtype != np.uint8:
      self._log(logging.DEBUG, f"Casting displayed image from {img.dtype} "
                               f"to uint8")
      if np.max(img) > 255:
        factor = min((i for i in range(1, 10) if np.max(img) / 2 ** i < 256))
        img = (img / 2 ** factor).astype(np.uint8)
      else:
        img = img.astype(np.uint8)

    # Calling the right prepare method
    if self._backend == 'cv2':
      self._update_cv2(img)
    elif self._backend == 'mpl':
      self._update_mpl(img)

  def finish(self) -> None:
    """Calls the right finish method depending on the chosen backend."""

    self._log(logging.INFO, "Closing the displayer window")
    if self._backend == 'cv2':
      self._finish_cv2()
    elif self._backend == 'mpl':
      self._finish_mpl()

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
      factor = min((i for i in range(2, 10) if img.shape[0] / i <= 480
                    and img.shape[1] / i <= 640))
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

  def _log(self, level: int, msg: str) -> None:
    """Sends a log message to the logger.

    Args:
      level: The logging level, as an :obj:`int`.
      msg: The message to log, as a :obj:`str`.
    """

    if self._logger is None:
      self._logger = logging.getLogger(f"{current_process().name}.Displayer")

    self._logger.log(level, msg)
