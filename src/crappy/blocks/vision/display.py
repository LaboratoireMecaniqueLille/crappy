# coding: utf-8

from typing import Literal
from time import time
import numpy as np
import logging
from math import ceil, log2
from itertools import chain
from collections.abc import Sequence

from .block import VisionBlock
from ..._global import OptionalModule
from ...tool.camera_config import Overlay

plt = OptionalModule('matplotlib.pyplot', lazy_import=True)

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class ImageDisplayer(VisionBlock):
  """Displays an image stream and optional overlays in a control window.

  This Block receives images from exactly one upstream VisionBlock through an
  input :class:`~crappy.links.ImageLink` and supports no output ImageLink. It
  is intended for live monitoring rather than high-rate or full-resolution
  image inspection: displayed images are limited to 640x480 pixels and updates
  are capped by ``framerate``. When the consumer falls behind, it displays the
  newest available image and skips intermediate frames.

  OpenCV and Matplotlib display backends are supported. If no backend is
  selected, OpenCV is preferred when available and Matplotlib is used as a
  fallback. Images whose dtype is not ``uint8`` are converted before display
  and large positive integer ranges are reduced by a power-of-two scale factor.

  Regular input :class:`~crappy.links.Link` objects can provide overlays under
  the reserved ``'overlay'`` label. Each value must be an iterable containing
  :class:`~crappy.tool.camera_config.Overlay` objects or :obj:`None`
  placeholders. The latest valid iterable from each Link is retained and drawn
  on subsequent images. Sending an empty iterable clears that Link's overlays,
  and malformed overlay values are ignored with a warning.

  After displaying an image, the Block sends its timestamp, unique image ID,
  and complete metadata through regular output Links under ``'t(s)'``,
  ``'img_index'``, and ``'meta'``. This makes downstream actions depend on
  images that were actually displayed rather than merely acquired.

  Unlike :class:`~crappy.blocks.camera_processes.Displayer`, which is managed
  internally by the older :class:`~crappy.blocks.Camera`, this class is an
  independent Block that can be connected anywhere in a VisionBlock pipeline.

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
    """Sets the display backend, rate, and standard Block options.

    Args:
      title: Text displayed in the window title bar. If omitted, a unique title
        based on the number of instantiated ImageDisplayers is generated.
      framerate: Maximum display update frequency. The achieved rate can be
        lower, but will never exceed this value. It must be strictly positive
        and no greater than ``freq`` when ``freq`` is set.
      backend: Display backend, either ``'cv2'`` for OpenCV or ``'mpl'`` for
        Matplotlib. If omitted, OpenCV is preferred and Matplotlib is used as a
        fallback.
      display_freq: If :obj:`True`, periodically reports the achieved display
        frequency.
      debug: If :obj:`True`, displays all log messages, including
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
      freq: Target frequency for checking overlays and images. If :obj:`None`,
        loops as fast as possible. It bounds how often the requested display
        framerate can be reached.
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
    self._overlay_buffer: dict[str, Sequence[Overlay | None]] = dict()
    self._last_warn: float = -float('inf')

  def __new__(cls, *args, **kwargs):
    """Allocates an instance and advances the automatic-title counter."""

    cls._count += 1
    return super().__new__(cls)

  def prepare(self) -> None:
    """Validates the ImageLink topology and opens the display window.

    The selected backend is initialized before the input shared image buffer is
    attached by :class:`VisionBlock`.

    Raises:
      IOError: If the Block does not have exactly one input ImageLink or has an
        output ImageLink.
    """

    # Ensuring Link consistency
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
    """Displays the newest eligible image with the latest overlays.

    Overlay inputs are consumed first so their newest values are retained even
    during loops skipped by the display-rate limit. Once the next display time
    is reached, the newest image is copied from the input ImageLink, converted
    to ``uint8`` when necessary, decorated with every retained overlay, and
    displayed by the selected backend.

    A dictionary containing ``'t(s)'``, ``'img_index'``, and ``'meta'`` is sent
    through regular output Links after a successful display update. If no new
    image is available or the framerate interval has not elapsed, the method
    returns without updating the window.

    Raises:
      RuntimeError: If image metadata is unavailable or does not contain
        ``'t(s)'`` and ``'ImageUniqueID'``.
    """

    # Update overlay buffer with latest overlays received from upstream Blocks
    # Not using regular Block methods as we need to differentiate overlays here
    for link in self.inputs:
      if 'overlay' in (data := link.recv_last()):
        try:
          overlays = tuple(data['overlay'])
        except (Exception,):
          if time() - self._last_warn > 2:
            self.log(logging.WARNING, f"Ignoring invalid overlay data received"
                                      f" from Link {link.name}: expected an "
                                      f"iterable of Overlay objects")
            self._last_warn = time()
          continue

        if not all(overlay is None or isinstance(overlay, Overlay)
                   for overlay in overlays):
          if time() - self._last_warn > 2:
            self.log(logging.WARNING, f"Ignoring invalid overlay data received"
                                      f" from Link {link.name}: expected only "
                                      f"Overlay objects or None placeholders")
            self._last_warn = time()
          continue

        self._overlay_buffer[link.name] = overlays

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

    # Handles to the received data
    metadata = self.last_received[upd_link].metadata
    if metadata is None:
      raise RuntimeError("At that point, the image metadata should not be "
                         "empty")
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

    # Drawing the latest known overlays
    for overlay in chain.from_iterable(self._overlay_buffer.values()):
      if overlay is not None:
        self.log(logging.DEBUG, f"Drawing {overlay} on top of the image to "
                                "display")
        overlay.draw(img)

    # Calling the right update method
    if self._backend == 'cv2':
      self._update_cv2(img)
    elif self._backend == 'mpl':
      self._update_mpl(img)

    # Sending information on the image through regular Links
    if 't(s)' not in metadata or 'ImageUniqueID' not in metadata:
      raise RuntimeError("At that point, 't(s)' and 'ImageUniqueID' should be "
                         "in the metadata dictionary")
    self.send({'t(s)': metadata['t(s)'],
               'img_index': metadata['ImageUniqueID'],
               'meta': metadata})

    # If requested, displays the FPS of the image display
    if self.display_freq:
      self._print_freq(img_handled=True)

  def finish(self) -> None:
    """Closes the display window and releases the input image buffer."""

    # Closing the Displayer window
    self.log(logging.INFO, "Closing the displayer window")
    if self._backend == 'cv2':
      self._finish_cv2()
    elif self._backend == 'mpl':
      self._finish_mpl()

    super().finish()

  def _prepare_cv2(self) -> None:
    """Creates a resizable OpenCV display window."""

    try:
      flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    except AttributeError:
      flags = cv2.WINDOW_NORMAL
    cv2.namedWindow(self._title, flags)

  def _prepare_mpl(self) -> None:
    """Enables interactive Matplotlib mode and creates a Figure."""

    plt.ion()
    self._fig, self._ax = plt.subplots()

  def _update_cv2(self, img: np.ndarray) -> None:
    """Downscales an image when needed and displays it with OpenCV.

    Args:
      img: ``uint8`` image to display.
    """

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
    """Subsamples an image when needed and displays it with Matplotlib.

    Args:
      img: ``uint8`` image to display.
    """

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
    """Destroys the OpenCV display window."""

    if self._title is not None:
      cv2.destroyWindow(self._title)

  def _finish_mpl(self) -> None:
    """Closes the Matplotlib Figure when it was created."""

    if self._fig is not None:
      plt.close(self._fig)
