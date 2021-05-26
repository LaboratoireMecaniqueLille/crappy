# coding: utf-8

from time import time, sleep
from .camera import Camera
from .._global import OptionalModule
from threading import Thread, RLock

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class Picamera_webcam(Camera):
  """Class for reading images from a PiCamera

  The Picamera_webcam Camera block is meant for reading images from a Picamera.
  It relies on openCV and the V4L2 API just like the Webcam block, hence its
  name. Faster than the Picamera block, but only the size of the image can be
  tuned. It is also preferable to set the size in the args of the block, as
  openCV is quite buggy when tuning it in the graphical interface.

  Warning:
    Only works on Raspberry Pi !
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

    # Thread and lock definition
    self._lock = RLock()
    self._thread = Thread(target=self._grab_frame)

  def open(self, **kwargs: any) -> None:
    """Sets the settings to their default values and starts buffering images"""

    # Setting the video stream
    if self._cap is not None:
      self._cap.release()
    self._cap = cv2.VideoCapture(cv2.CAP_V4L2)

    # Setting the image parameters
    for k in kwargs:
      assert k in self.available_settings, \
        str(self) + "Unexpected kwarg: " + str(k)
    self.set_all(**kwargs)

    # Starting the auxiliary thread
    self._thread.start()

  def get_image(self) -> [float, any]:
    """Reads an image from the PiCamera using V4L2

    Returns:
      The timeframe and the image
    """

    t = time()

    # Reading the stream
    with self._lock:
      ret, frame = self._cap.read()
    if not ret:
      raise IOError("Error reading the camera")

    # Converting to black and white if necessary
    if self.channels == 1:
      return t, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
      return t, frame

  def close(self) -> None:
    """Releases the resources and joins the thread"""

    # Joining the auxiliary thread and closing the stream
    self._stop = True
    self._thread.join()
    if self._cap is not None:
      self._cap.release()
    self._cap = None

  def _grab_frame(self) -> None:
    """
    This thread is meant for preventing the accumulation of frames in the
    video stream. Every 0.01s it tries to grab frames from the stream ; if
    one is available it is grabbed and otherwise nothing happens. This way
    only the last captured frame can be accessed by the ``get_image`` method.
    """

    while not self._stop:
      with self._lock:
        self._cap.grab()
      sleep(0.01)

  def _get_width(self) -> int:
    with self._lock:
      return self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)

  def _get_height(self) -> int:
    with self._lock:
      return self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

  def _set_width(self, width: float) -> None:
    # The Picamera only accepts width that are multiples of 32
    with self._lock:
      self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 32 * (width // 32))

  def _set_height(self, height: float) -> None:
    # The Picamera only accepts heights that are multiples of 32
    with self._lock:
      self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 32 * (height // 32))
