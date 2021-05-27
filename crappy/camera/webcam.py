# coding: utf-8

from time import time
from .camera import Camera
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class Webcam(Camera):
  """Camera class for webcams, read using opencv."""

  def __init__(self):
    Camera.__init__(self)
    self.name = "webcam"
    self.cap = None
    # No sliders for the camera: they usually only allow a few resolutions
    self.add_setting("width", self._get_w, self._set_w, (1, 1920))
    self.add_setting("height", self._get_h, self._set_h, (1, 1080))
    self.add_setting("channels", limits={1: 1, 3: 3}, default=1)

  def _get_w(self):
    return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

  def _get_h(self):
    return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

  def _set_w(self, width):
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

  def _set_h(self, height):
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  def open(self, numdevice=0, **kwargs):
    self.numdevice = numdevice
    if self.cap:
      self.cap.release()
    self.cap = cv2.VideoCapture(self.numdevice)
    for k in kwargs:
      assert k in self.available_settings, str(self) + \
                                           "Unexpected kwarg: " + str(k)
    self.set_all(**kwargs)

  def get_image(self):
    ret, frame = self.cap.read()
    t = time()
    if not ret:
      print("Error reading the camera")
      raise IOError
    if self.channels == 1:
      return t, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
      return t, frame  # [:, :, [2, 1, 0]]

  def close(self):
    if self.cap:
      self.cap.release()
    self.cap = None
