# coding: utf-8

from time import time, sleep
from typing import Any, Optional
import numpy as np
from threading import Thread, RLock
import logging
from warnings import warn

from .meta_camera import Camera
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")

try:
  from picamera import PiCamera as PiCameraRPi
  from picamera.array import PiRGBArray
except (ModuleNotFoundError, ImportError, OSError):
  PiCameraRPi = OptionalModule("picamera")
  PiRGBArray = OptionalModule("picamera")

picamera_iso = [0, 100, 200, 320, 400, 500, 640, 800]


class RaspberryPiCamera(Camera):
  """Class for reading images from a Raspberry Pi Camera, using the legacy
  :mod:`picamera2` module.

  The RaspberryPiCamera Camera is meant for reading images from a Raspberry Pi
  Camera. It uses the :mod:`picamera` module for capturing images, and
  :mod:`cv2` for converting BGR images to black and white.

  It can read images from the PiCamera V1, V2 and HQ models.

  Warning:
    This class is a legacy object, the Camera
    :class:`~crappy.camera.RaspberryPiCamera2` should be used instead. This
    class can only be used on the "Buster" OS, or the "Bullseye" OS with legacy
    camera mode enabled.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Picamera* to *RaspberryPiCamera*
  """

  def __init__(self) -> None:
    """Instantiates the available settings."""

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    super().__init__()

    self._frame_grabber: Optional[Thread] = None
    self._capture = None
    self._cam = None

    self.log(logging.INFO, "Opening the connection to the camera")
    self._cam = PiCameraRPi()

    # Settings definition
    self.add_scale_setting('Width', 1, 3280, self._get_width,
                           self._set_width, 1280)
    self.add_scale_setting('Height', 1, 2464, self._get_height,
                           self._set_height, 720)
    self.add_scale_setting('Iso (0 for auto)', 0, 800, self._get_iso,
                           self._set_iso, 0)
    self.add_scale_setting('Brightness', 0, 100, self._get_brightness,
                           self._set_brightness, 50)
    self.add_scale_setting('Contrast', -100, 100, self._get_contrast,
                           self._set_contrast, 0)
    self.add_scale_setting('Saturation', -100, 100, self._get_saturation,
                           self._set_saturation, 0)
    self.add_scale_setting('Shutter speed (0 for auto)', 0, 30,
                           self._get_shutter_speed, self._set_shutter_speed, 0)
    self.add_bool_setting('Black_and_white', self._get_black_white,
                          self._set_black_white, True)
    self.add_scale_setting('Crop: X offset', 0.0, 1.0, self._get_crop_x_offset,
                           self._set_crop_x_offset, 0.0)
    self.add_scale_setting('Crop: Y offset', 0.0, 1.0, self._get_crop_y_offset,
                           self._set_crop_y_offset, 0.0)
    self.add_scale_setting('Crop: width', 0.0, 1.0, self._get_crop_width,
                           self._set_crop_width, 1.0)
    self.add_scale_setting('Crop: height', 0.0, 1.0, self._get_crop_height,
                           self._set_crop_height, 1.0)

    self._frame_grabber = Thread(target=self._grab_frame)
    self._lock = RLock()
    self._frame = None
    self._stop = False
    self._started = False
    self._stream = None

  def open(self, **kwargs: Any) -> None:
    """Sets the settings to their default values and starts the image
    acquisition thread."""

    self.set_all(**kwargs)

    # Starting the video stream
    self._capture = PiRGBArray(self._cam, (self._get_width(),
                                           self._get_height()))
    self.log(logging.INFO, "Starting the frame stream")
    self._stream = self._cam.capture_continuous(self._capture, format='bgr',
                                                use_video_port=True)

    self.log(logging.INFO, "Starting the frame grabber thread")
    self._frame_grabber.start()
    sleep(1)
    self._started = True

  def get_image(self) -> tuple[float, np.ndarray]:
    """Simply returns the last image in the acquisition buffer.

    The captured image is in GBR format, and converted into black and white if
    needed.

    Returns:
      The timeframe and the image.
    """

    t = time()
    with self._lock:
      output = self._frame
    if self.Black_and_white:
      output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    return t, output

  def close(self) -> None:
    """Joins the image acquisition thread, and closes the stream and the
    :class:`picamera.PiCamera` object."""

    self._stop = True
    if self._frame_grabber is not None:
      self.log(logging.INFO, "Stopping the frame grabber thread")
      self._frame_grabber.join(0.2)
      if self._frame_grabber.is_alive():
        self.log(logging.WARNING, "The frame grabber thread didn't stop "
                                  "properly !")

    if self._capture is not None:
      self.log(logging.INFO, "Stopping the frame stream")
      self._capture.close()

    if self._cam is not None:
      self.log(logging.INFO, "Opening the connection to the camera")
      self._cam.close()

  def _stop_stream(self) -> None:
    """Stops the video stream. Called before changing the image size."""

    if not self._started:
      return

    self._stop = True
    self.log(logging.INFO, "Stopping the frame grabber thread")
    self._frame_grabber.join()
    self.log(logging.INFO, "Stopping the frame stream")
    self._capture.close()

  def _restart_stream(self) -> None:
    """Restarts the video stream. Called after a change in the image size."""

    if not self._started:
      return

    self._stop = False
    self._frame_grabber = Thread(target=self._grab_frame)
    self._capture = PiRGBArray(self._cam, (self._get_width(),
                                           self._get_height()))
    self.log(logging.INFO, "Starting the frame stream")
    self._stream = self._cam.capture_continuous(self._capture, format='bgr',
                                                use_video_port=True)

    self.log(logging.INFO, "Starting the frame grabber thread")
    self._frame_grabber.start()
    sleep(1)

  def _grab_frame(self) -> None:
    """Target of a thread for grabbing the last image in the video stream and
    putting it in a buffer."""

    for frame in self._stream:
      with self._lock:
        self.log(logging.DEBUG, "Got new frame from stream")
        self._frame = frame.array
      self._capture.truncate(0)
      if self._stop:
        break

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
    # The Raspberry Pi Camera only accepts width that are multiples of 32
    self._stop_stream()
    self._cam.resolution = (32 * (width // 32), self._get_height())
    self._restart_stream()

  def _set_height(self, height: float) -> None:
    # The Raspberry Pi Camera only accepts heights that are multiples of 32
    self._stop_stream()
    self._cam.resolution = (self._get_width(), 32 * (height // 32))
    self._restart_stream()

  def _set_iso(self, iso: float) -> None:
    # The Raspberry Pi Camera only accepts a limited range of iso values
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
