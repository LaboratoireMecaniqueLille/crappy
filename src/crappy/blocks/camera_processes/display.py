# coding: utf-8

from threading import Thread
from math import log2, ceil
import numpy as np
from typing import Optional
from collections.abc import Iterable
from time import time, sleep
import logging
import logging.handlers

from .camera_process import CameraProcess
from ..._global import OptionalModule
from ...tool.camera_config import Overlay

plt = OptionalModule('matplotlib.pyplot', lazy_import=True)

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class Displayer(CameraProcess):
  """This :class:`~crappy.blocks.camera_processes.CameraProcess` can display 
  images acquired by a :class:`~crappy.blocks.Camera` Block in a dedicated 
  window.
  
  It is meant to serve as a control or validation feature, its resolution is
  thus limited to `640x480` and it should not be used at high framerates. On
  top of the displayed image, it can also draw 
  :class:`~crappy.tool.camera_config.config_tools.Overlay` objects sent by
  other :class:`~crappy.blocks.camera_processes.CameraProcess`. This way, the
  user can for example visualize the spots being tracked by the
  :class:`~crappy.blocks.VideoExtenso` Block in real time.

  The images can be displayed using two different backends : either using
  :mod:`cv2` (OpenCV), or using :mod:`matplotlib`. OpenCV is by far the fastest
  and most convenient.

  .. versionadded:: 2.0.0
  """

  def __init__(self,
               title: str,
               framerate: float,
               backend: Optional[str] = None) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      title: The name of the Displayer window, that will be displayed on the
        window border.
      framerate: The target framerate for the display. The actual achieved
        framerate might be lower, but never greater than this value.
      backend: The module to use for displaying the images. Can be either
        ``'cv2'`` or ``'mpl'``, to use respectively :mod:`cv2` or
        :mod:`matplotlib`.
    """

    # The thread must be initialized later for compatibility with Windows
    self._overlay_thread: Optional[Thread] = None
    self._overlay: Iterable[Overlay] = list()
    self._stop_thread = False

    super().__init__()

    self._title = title
    self._framerate = framerate

    # Selecting the backend if no backend was specified
    if backend is None:
      if not isinstance(cv2, OptionalModule):
        self._backend = 'cv2'
      else:
        try:
          _ = plt.Figure
          self._backend = 'mpl'
        except RuntimeError:
          raise ModuleNotFoundError("Neither opencv-python nor matplotlib "
                                    "could be imported, no backend found for "
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

  def __del__(self) -> None:
    """On exit, ensuring that the :obj:`~threading.Thread` in charge of
    grabbing the :class:`~crappy.tool.camera_config.config_tools.Overlay` to
    display has stopped, otherwise stopping it."""

    if self._overlay_thread is not None and self._overlay_thread.is_alive():
      self._stop_thread = True
      try:
        self._overlay_thread.join(0.05)
      except RuntimeError:
        pass

  def init(self) -> None:
    """Starts the :obj:`~threading.Thread` for grabbing the
    :class:`~crappy.tool.camera_config.config_tools.Overlay` to display, and
    initializes the Displayer window."""

    # Instantiating and starting the Thread for grabbing the Overlays
    self.log(logging.INFO, "Instantiating the thread for getting the Overlays"
                           " to display")
    self._overlay_thread = Thread(target=self._thread_target)
    self.log(logging.INFO, "Starting the thread for getting the Overlays to "
                           "display")
    self._overlay_thread.start()

    # Preparing the Displayer window
    self.log(logging.INFO, f"Opening the displayer window with the backend "
                           f"{self._backend}")
    if self._backend == 'cv2':
      self._prepare_cv2()
    elif self._backend == 'mpl':
      self._prepare_mpl()

  def _get_data(self) -> bool:
    """Method similar to the one of the parent class, except it also ensures 
    that the achieved framerate stays within the limit specified by the user.
    
    Returns:
      :obj:`True` in case a frame was acquired and needs to be handled, or
      :obj:`False` if no frame was grabbed and nothing should be done.
    """

    # Acquiring the Lock to avoid conflicts with other CameraProcesses
    with self._lock:

      # In case there's no frame grabbed yet
      if 'ImageUniqueID' not in self._data_dict:
        return False

      # In case the frame in buffer was already handled during a previous loop,
      # or it's too early to grab a new frame because of the target framerate
      if self._data_dict['ImageUniqueID'] == self.metadata['ImageUniqueID'] \
          or time() - self._last_upd < 1 / self._framerate:
        return False

      # Copying the metadata
      self.metadata = self._data_dict.copy()
      self._last_upd = time()

      self.log(logging.DEBUG, f"Got new image to process with id "
                              f"{self.metadata['ImageUniqueID']}")

      # Copying the frame
      np.copyto(self.img,
                np.frombuffer(self._img_array.get_obj(),
                              dtype=self._dtype).reshape(self._shape))

    return True

  def loop(self) -> None:
    """This method grabs the latest frame, casts it to 8 bits if necessary,
    and updates the Displayer window to draw it.
    
    It also draws the latest received 
    :class:`~crappy.tool.camera_config.config_tools.Overlay` on top of the
    displayed frame.
    """

    # Casting the image to uint8 if it's not already in this format
    if self.img.dtype != np.uint8:
      self.log(logging.DEBUG, f"Casting displayed image from "
                              f"{self.img.dtype} to uint8")
      if max_pix := int(np.max(self.img)) > 255:
        factor = max(ceil(log2(max_pix + 1) - 8), 0)
        img = (self.img / 2 ** factor).astype(np.uint8)
      else:
        img = self.img.astype(np.uint8)
    else:
      img = self.img.copy()

    # Drawing the latest known overlay
    for overlay in self._overlay:
      if overlay is not None:
        self.log(logging.DEBUG, f"Drawing {overlay} on top of the image to "
                                "display")
        overlay.draw(img)

    # Calling the right update method
    if self._backend == 'cv2':
      self._update_cv2(img)
    elif self._backend == 'mpl':
      self._update_mpl(img)

  def finish(self) -> None:
    """Closes the Displayer window and stops the :obj:`~threading.Thread`
    grabbing the :class:`~crappy.tool.camera_config.config_tools.Overlay`"""

    # Closing the Displayer window
    self.log(logging.INFO, "Closing the displayer window")
    if self._backend == 'cv2':
      self._finish_cv2()
    elif self._backend == 'mpl':
      self._finish_mpl()

    # Stooping the Thread grabbing the Overlay to draw
    if self._overlay_thread is not None and self._overlay_thread.is_alive():
      self._stop_thread = True
      try:
        self._overlay_thread.join(0.05)
      except RuntimeError:
        self.log(logging.WARNING, "Thread for receiving the Overlay did not "
                                  "stop as expected")

  def _thread_target(self) -> None:
    """This method is the target to the :obj:`~threading.Thread` in charge of
    grabbing the :class:`~crappy.tool.camera_config.config_tools.Overlay` to
    draw on top of the displayed image.
    
    It repeatedly polls the :obj:`~multiprocessing.Connection` through which
    the Overlays are received, and stores the last received Overlays.
    """

    # Looping until the entire CameraProcess is told to stop, or the 
    # _stop_thread flag is raised
    while not self._stop_event.is_set() and not self._stop_thread:

      # Receiving the latest Overlay to draw
      overlay = None
      while self._to_draw_conn.poll():
        overlay = self._to_draw_conn.recv()

      # Saving the received Overlay
      if overlay is not None:
        self.log(logging.DEBUG, f"Received overlay to display: {overlay}")
        self._overlay = overlay

      # To avoid spamming the CPU in vain
      else:
        sleep(0.001)

    self.log(logging.INFO, "Thread for receiving the Overlays ended")

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
