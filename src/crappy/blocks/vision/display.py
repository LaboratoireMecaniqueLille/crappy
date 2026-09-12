# coding: utf-8

from typing import Literal
from time import time
import numpy as np
import logging
from math import ceil, log2

from .block import VisionBlock
from ..._global import OptionalModule

plt = OptionalModule('matplotlib.pyplot', lazy_import=True)

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class ImageDisplayer(VisionBlock):
  """This :class:`~crappy.blocks.vision.VisionBlock` can display images
  received from upstream VisionBlocks Block in a dedicated window.

  It is meant to serve as a control or validation feature, its resolution is
  thus limited to `640x480` and it should not be used at high framerates.

  The images can be displayed using two different backends : either using
  :mod:`cv2` (OpenCV), or using :mod:`matplotlib`. OpenCV is by far the fastest
  and most convenient.

  .. versionadded:: 2.1.0
  """

  _count: int = 0

  def __init__(self,
               title: str | None = None,
               framerate: float = 5.0,
               backend: Literal['cv2', 'mpl'] | None = None,
               display_freq: bool = False,
               debug: bool | None = False,
               freq: float | None = 100) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      title: The name of the Displayer window, that will be displayed on the
        window border, as a :obj:`str`. If not provided, a name will be
        automatically assigned based on the number of already instantiated
        Displayers in the script.
      framerate: The target framerate for the display, as a :obj:`float`. The
        actual achieved framerate might be lower, but never greater than this
        value.
      backend: The module to use for displaying the images. Can be either
        ``'cv2'`` or ``'mpl'``, to use respectively :mod:`cv2` or
        :mod:`matplotlib`.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.
    """

    super().__init__(img_shape=None,
                     img_dtype=None,
                     display_freq=display_freq,
                     debug=debug,
                     freq=freq)

    # Validate title before setting it
    if title is not None:
      if not isinstance(title, str) or not title:
        raise ValueError("The provided Displayer title must be a non-empty "
                         "string")
      self._title: str = title
    else:
      self._title: str = f"Displayer {self._count}"

    # Validate framerate before setting it
    if ((not isinstance(framerate, int) and
        not isinstance(framerate, float))
        or framerate <= 0):
      raise ValueError("framerate must be a strictly positive float or int")
    if freq is not None and framerate > freq:
      raise ValueError("The displayer framerate is by nature inferior to the "
                       "freq!")
    self._framerate: float = framerate

    # Validate backend before setting it
    if backend is None:
      if not isinstance(cv2, OptionalModule):
        self._backend: str = 'cv2'
      else:
        try:
          _ = plt.Figure
          self._backend: str = 'mpl'
        except RuntimeError:
          raise ModuleNotFoundError("Neither opencv-python nor matplotlib "
                                    "could be imported, no backend found for "
                                    "displaying the images")
    elif backend in ('cv2', 'mpl'):
      self._backend: str = backend
    else:
      raise ValueError("The backend argument should be either 'cv2' or 'mpl'!")

    # Setting other attributes
    self._ax = None
    self._fig = None
    self._last_upd: float = float('-inf')

  def __new__(cls, *args, **kwargs):
    """When instantiating a new displayer, increments the Displayer counter."""

    cls._count += 1
    return super().__new__(cls)

  def prepare(self) -> None:
    """Initializes the Displayer window"""

    # Ensuring Link consistency
    if self.inputs:
      raise IOError("This Block does not accept input Links")
    if self.img_outputs:
      raise IOError("This VisionBlock does not support output ImageLink")
    if not self.img_inputs:
      raise IOError("This VisionBlock is useless without an input ImageLink")
    if len(self.img_inputs) != 1:
      raise IOError("This VisionBlock requires exactly one input ImageLink")

    # Preparing the Displayer window
    if self._backend == 'cv2':
      self._prepare_cv2()
    elif self._backend == 'mpl':
      self._prepare_mpl()

    super().prepare()

  def loop(self) -> None:
    """This method grabs the latest frame, casts it to 8 bits if necessary,
    and updates the Displayer window to draw it.

    In addition, a message containing information on each displayed image is
    sent through the output :class:`~crappy.links.Link` if any.
    """

    # Enforce framerate by skipping loops
    if time() - self._last_upd < 1 / self._framerate:
      self.log(logging.DEBUG, "Too early to loop, to achieve the desired "
                              "framerate")
      # If requested, displays the FPS of the image display
      if self.display_freq:
        self._print_freq(img_handled=False)
      return
    # Update last received time
    self._last_upd = time()

    # Nothing to do if no new image was received
    if not (upd_links := self.receive_imgs()):
      self.log(logging.DEBUG, "No new image received during this loop")
      # If requested, displays the FPS of the image display
      if self.display_freq:
        self._print_freq(img_handled=False)
      return
    # Get the ImageLink name
    upd_link, = upd_links

    img = self.last_received[upd_link].img

    # Casting the image to uint8 if it's not already in this format
    if img.dtype != np.uint8:
      self.log(logging.DEBUG, f"Casting displayed image from "
                              f"{img.dtype} to uint8")
      if (max_pix := int(np.max(img))) > 255:
        factor = max(ceil(log2(max_pix + 1) - 8), 0)
        img = (img / 2 ** factor).astype(np.uint8)
      else:
        img = img.astype(np.uint8)
    else:
      img = img.copy()

    # Calling the right update method
    if self._backend == 'cv2':
      self._update_cv2(img)
    elif self._backend == 'mpl':
      self._update_mpl(img)

    # Sending information on the image through regular Links
    if self.last_received[upd_link].metadata is None:
      raise RuntimeError("At that point, the image metadata should not be "
                         "empty")
    if ('t(s)' not in self.last_received[upd_link].metadata
        and 'ImageUniqueID' not in self.last_received[upd_link].metadata):
      raise RuntimeError("At that point, 't(s)' and 'ImageUniqueID' should be "
                         "in the metadata dictionary")
    self.send({
      't(s)': self.last_received[upd_link].metadata['t(s)'],
      'img_index': self.last_received[upd_link].metadata['ImageUniqueID'],
      'meta': self.last_received[upd_link].metadata})

    # If requested, displays the FPS of the image display
    if self.display_freq:
      self._print_freq(img_handled=True)

  def finish(self) -> None:
    """Closes the Displayer window."""

    # Closing the Displayer window
    self.log(logging.INFO, "Closing the displayer window")
    if self._backend == 'cv2':
      self._finish_cv2()
    elif self._backend == 'mpl':
      self._finish_mpl()

    super().finish()

  def _prepare_cv2(self) -> None:
    """Instantiates the display window of :mod:`cv2`."""

    try:
      flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    except AttributeError:
      flags = cv2.WINDOW_NORMAL
    cv2.namedWindow(self._title, flags)

  def _prepare_mpl(self) -> None:
    """Creates a :mod:`matplotlib` Figure."""

    plt.ion()
    self._fig, self._ax = plt.subplots()

  def _update_cv2(self, img: np.ndarray) -> None:
    """Reshapes the image to a maximum shape of 640x480 and displays it in
    :mod:`cv2`."""

    if img.shape[0] > 480 or img.shape[1] > 640:
      factor = min(480 / img.shape[0], 640 / img.shape[1])
      self.log(logging.DEBUG,
               f"Reshaping displayed image from {img.shape} to "
               f"{int(img.shape[1] * factor), int(img.shape[0] * factor)}")
      img = cv2.resize(img, (int(img.shape[1] * factor),
                             int(img.shape[0] * factor)))

    self.log(logging.DEBUG, "Displaying the image")
    cv2.imshow(self._title, img)
    cv2.waitKey(1)

  def _update_mpl(self, img: np.ndarray) -> None:
    """Reshapes the image to a dimension inferior or equal to 640x480 and
    displays it in :mod:`matplotlib`."""

    if img.shape[0] > 480 or img.shape[1] > 640:
      factor = max(ceil(img.shape[0] / 480), ceil(img.shape[1] / 640))
      self.log(logging.DEBUG,
               f"Reshaping the displayed image from {img.shape} to "
               f"{(img.shape[0] / factor, img.shape[1] / factor)}")
      img = img[::factor, ::factor]

    self._ax.clear()
    self.log(logging.DEBUG, "Displaying the image")
    self._ax.imshow(img, cmap='gray')
    plt.pause(0.001)
    plt.show()

  def _finish_cv2(self) -> None:
    """Destroys the opened :mod:`cv2` window."""

    if self._title is not None:
      cv2.destroyWindow(self._title)

  def _finish_mpl(self) -> None:
    """Destroys the opened :mod:`matplotlib` window."""

    if self._fig is not None:
      plt.close(self._fig)
