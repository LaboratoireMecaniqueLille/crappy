# coding: utf-8

from time import time
from typing import Tuple, Any
from .camera import Camera
from .._global import OptionalModule
import numpy as np

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")

try:
  from picamera import PiCamera
except (ModuleNotFoundError, ImportError):
  PiCamera = OptionalModule("picamera")

picamera_iso = [0, 100, 200, 320, 400, 500, 640, 800]


class Picamera(Camera):
  """Class for reading images from a PiCamera.

  The Picamera Camera block is meant for reading images from a Picamera.
  It uses the :mod:`picamera` module for capturing images, and :mod:`cv2` for
  converting bgr images to black and white. The framerate is not optimal, but
  it allows to tune a wide variety of image properties.

  Warning:
    Only works on Raspberry Pi !
  """

  def __init__(self) -> None:
    """Instantiates the available settings."""

    Camera.__init__(self)
    self._cam = PiCamera()
    self.name = "picamera"

    # Settings definition
    self.add_setting("Width", self._get_width, self._set_width, (1, 3280),
                     default=1280)
    self.add_setting("Height", self._get_height, self._set_height, (1, 2464),
                     default=720)
    self.add_setting("Iso (0 for auto)", self._get_iso, self._set_iso,
                     (0, 800), default=0)
    self.add_setting("Brightness", self._get_brightness, self._set_brightness,
                     (0, 100), default=50)
    self.add_setting("Contrast", self._get_contrast, self._set_contrast,
                     (-100, 100), default=0)
    self.add_setting("Saturation", self._get_saturation, self._set_saturation,
                     (-100, 100), default=0)
    self.add_setting("Shutter speed (0 for auto)", self._get_shutter_speed,
                     self._set_shutter_speed, (0, 30), default=0)
    self.add_setting("Black_and_white", self._get_black_white,
                     self._set_black_white, False, default=True)
    self.add_setting("Crop: X offset", self._get_crop_x_offset,
                     self._set_crop_x_offset, (0.0, 1.0), default=0.0)
    self.add_setting("Crop: Y offset", self._get_crop_y_offset,
                     self._set_crop_y_offset, (0.0, 1.0), default=0.0)
    self.add_setting("Crop: width", self._get_crop_width, self._set_crop_width,
                     (0.0, 1.0), default=1.0)
    self.add_setting("Crop: height", self._get_crop_height,
                     self._set_crop_height, (0.0, 1.0), default=1.0)

  def open(self, **kwargs: Any) -> None:
    """Sets the settings to their default values."""

    for k in kwargs:
      assert k in self.available_settings, \
        str(self) + "Unexpected kwarg: " + str(k)
    self.set_all(**kwargs)

  def get_image(self) -> Tuple[float, np.ndarray]:
    """Uses the picamera capture method for capturing an image.

    The captured image is in bgr format, and converted into black and white if
    needed. Quite slow since the encoder has to initialize before each capture.

    Returns:
      The timeframe and the image
    """

    output = np.empty((self.Height, self.Width, 3), dtype=np.uint8)
    t = time()
    self._cam.capture(output, 'bgr', use_video_port=True)
    if self.Black_and_white:
      output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    return t, output

  def close(self) -> None:
    """Closes the :class:`picamera.PiCamera` object."""

    self._cam.close()

  def _get_width(self) -> int:
    return self._cam.resolution[0]

  def _get_height(self) -> int:
    return self._cam.resolution[1]

  def _get_iso(self) -> int:
    return self._cam.iso

  def _get_brightness(self) -> int:
    return self._cam.brightness

  def _get_contrast(self) -> int:
    return self._cam.contrast

  def _get_saturation(self) -> int:
    return self._cam.saturation

  def _get_shutter_speed(self) -> float:
    return self._cam.shutter_speed

  def _get_black_white(self) -> bool:
    return self._cam.color_effects == (128, 128)

  def _get_crop_x_offset(self) -> float:
    return self._cam.zoom[0]

  def _get_crop_y_offset(self) -> float:
    return self._cam.zoom[1]

  def _get_crop_width(self) -> float:
    return self._cam.zoom[2]

  def _get_crop_height(self) -> float:
    return self._cam.zoom[3]

  def _set_width(self, width: float) -> None:
    # The Picamera only accepts width that are multiples of 32
    self._cam.resolution = (32 * (width // 32), self._get_height())

  def _set_height(self, height: float) -> None:
    # The Picamera only accepts heights that are multiples of 32
    self._cam.resolution = (self._get_width(), 32 * (height // 32))

  def _set_iso(self, iso: float) -> None:
    # The Picamera only accepts a limited range of iso values
    self._cam.iso = min(picamera_iso, key=lambda x: abs(x - iso))

  def _set_brightness(self, brightness: float) -> None:
    self._cam.brightness = brightness

  def _set_contrast(self, contrast: float) -> None:
    self._cam.contrast = contrast

  def _set_saturation(self, saturation: float) -> None:
    self._cam.saturation = saturation

  def _set_shutter_speed(self, shutter_speed: float) -> None:
    self._cam.shutter_speed = shutter_speed

  def _set_black_white(self, boolean: bool) -> None:
    if boolean:
      self._cam.color_effects = (128, 128)
    else:
      self._cam.color_effects = None

  def _set_crop_x_offset(self, x_offset: float) -> None:
    self._cam.zoom = (x_offset, self._get_crop_y_offset(),
                     self._get_crop_width(), self._get_crop_height())

  def _set_crop_y_offset(self, y_offset: float) -> None:
    self._cam.zoom = (self._get_crop_x_offset(), y_offset,
                     self._get_crop_width(), self._get_crop_height())

  def _set_crop_width(self, width: float) -> None:
    self._cam.zoom = (self._get_crop_x_offset(), self._get_crop_y_offset(),
                     width, self._get_crop_height())

  def _set_crop_height(self, height: float) -> None:
    self._cam.zoom = (self._get_crop_x_offset(), self._get_crop_y_offset(),
                     self._get_crop_width(), height)
