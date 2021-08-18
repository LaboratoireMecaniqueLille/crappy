# coding: utf-8

from time import time
from typing import Tuple, Any
from .camera import Camera
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class Picamera_webcam(Camera):
  """Class for reading images from a PiCamera.

  The Picamera_webcam Camera block is meant for reading images from a Picamera.
  It relies on :mod:`cv2` and the V4L2 API just like the :ref:`Webcam` block,
  hence its name. The only differences with the :ref:`Webcam` block are that
  here the height and width must be multiples of 32, and the resolution limits
  are those of the Picamera.

  Note:
    Actually works with any camera, not only the Picamera.
  """

  def __init__(self) -> None:
    Camera.__init__(self)
    self.name = "picamera_webcam"
    self._cap = None
    self._stop = False

    # Settings definition
    self.add_setting("width", self._get_width, self._set_width, (1, 3280),
                     default=1280)
    self.add_setting("height", self._get_height, self._set_height, (1, 2464),
                     default=704)
    self.add_setting("channels", limits={1: 1, 3: 3}, default=1)

  def open(self, **kwargs: Any) -> None:
    """Sets the settings to their default values and starts buffering images.
    """

    # Setting the video stream
    if self._cap is not None:
      self._cap.release()
    self._cap = cv2.VideoCapture(cv2.CAP_V4L2)

    # Setting the image parameters
    for k in kwargs:
      assert k in self.available_settings, \
        str(self) + "Unexpected kwarg: " + str(k)
    self.set_all(**kwargs)

  def get_image(self) -> Tuple[float, Any]:
    """Reads an image from the PiCamera using V4L2.

    Returns:
      The timeframe and the image
    """

    t = time()

    # Reading the stream
    ret, frame = self._cap.read()
    if not ret:
      raise IOError("Error reading the camera")

    # Converting to black and white if necessary
    if self.channels == 1:
      return t, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
      return t, frame

  def close(self) -> None:
    """Releases the resources and joins the thread."""

    # Joining the auxiliary thread and closing the stream
    if self._cap is not None:
      self._cap.release()
    self._cap = None

  def _get_width(self) -> int:
    return self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)

  def _get_height(self) -> int:
    return self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

  def _set_width(self, width: float) -> None:
    # The Picamera only accepts width that are multiples of 32
    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 32 * (width // 32))

  def _set_height(self, height: float) -> None:
    # The Picamera only accepts heights that are multiples of 32
    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 32 * (height // 32))
