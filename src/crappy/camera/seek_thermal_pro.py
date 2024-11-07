# coding: utf-8

from typing import Any
import numpy as np
from time import time
import logging
from  warnings import warn

from .meta_camera import Camera
from .._global import OptionalModule

try:
  import usb.util
  import usb.core

  Seek_therm_usb_req = {'Write': usb.util.CTRL_OUT |
                        usb.util.CTRL_TYPE_VENDOR |
                        usb.util.CTRL_RECIPIENT_INTERFACE,
                        'Read': usb.util.CTRL_IN |
                        usb.util.CTRL_TYPE_VENDOR |
                        usb.util.CTRL_RECIPIENT_INTERFACE,
                        'Read_img': usb.util.CTRL_IN |
                        usb.util.CTRL_TYPE_STANDARD |
                        usb.util.CTRL_RECIPIENT_INTERFACE}

except (ModuleNotFoundError, ImportError):
  usb = OptionalModule("usb")

Seek_thermal_pro_vendor = 0x289D
Seek_thermal_pro_product = 0x0011

Seek_thermal_pro_commands = {'Read chip id': 0x36,
                             'Start get image transfer': 0x53,
                             'Get operation mode': 0x3D,
                             'Get image processing mode': 0x3F,
                             'Get firmware info': 0x4E,
                             'Get factory settings': 0x58,
                             'Set operation mode': 0x3C,
                             'Set image processing mode': 0x3E,
                             'Set firmware info features': 0x55,
                             'Set factory settings features': 0x56}

Seek_thermal_pro_dimensions = {'Width': 320,
                               'Height': 240,
                               'Raw width': 342,
                               'Raw height': 260}


class SeekThermalPro(Camera):
  """Class for reading images from the Seek Thermal Pro infrared camera.

  The SeekThermalPro Camera is meant for reading images from a Seek
  Thermal Pro infrared camera. It communicates over USB, and gets images by
  converting the received bytearrays into :mod:`numpy` arrays.

  Important:
    **Only for Linux users:** In order to drive the Seek Thermal Pro, the
    appropriate udev rule should be set. This can be done using the
    `udev_rule_setter` utility in ``crappy``'s `util` folder. It is also
    possible to add it manually by running:
    ::

      echo "SUBSYSTEM==\\"usb\\", ATTR{idVendor}==\\"289d\\", \
MODE=\\"0777\\\"" | sudo tee seek_thermal.rules > /dev/null 2>&1

    in a shell opened in ``/etc/udev/rules.d``.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Seek_thermal_pro* to *SeekThermalPro*
  """

  def __init__(self) -> None:
    """Selects the right USB device."""

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    super().__init__()

    self._dev = None
    self._calib = None
    self._dead_pixels = []

    # Listing all the matching USB devices
    devices = usb.core.find(find_all=True,
                            idVendor=Seek_thermal_pro_vendor,
                            idProduct=Seek_thermal_pro_product)
    devices = list(devices)

    # Making sure there's exactly one possible camera to read images from
    if len(devices) > 1:
      raise IOError("Several matching cameras found, impossible to "
                    "differentiate between them")
    elif len(devices) == 0:
      raise IOError("No matching camera found")
    else:
      self._dev = devices[0]

  def open(self) -> None:
    """Sets the USB communication and initializes the device."""

    # Setting the USB configuration on the camera
    try:
      self.log(logging.INFO, f"Setting configuration on USB device "
                             f"{self._dev}")
      self._dev.set_configuration()
    except usb.core.USBError:
      self.log(logging.ERROR,
               "An error occurred while setting the configuration of the USB"
               " device !\nYou may have to install the udev-rules for this "
               "USB device, this can be done using the udev_rule_setter "
               "utility in the util folder")
      raise

    # Initializing the camera by sending various commands to it
    self.log(logging.INFO, "Configuring the camera")
    self._write_data(Seek_thermal_pro_commands['Set operation mode'],
                     b'\x00\x00')
    self._write_data(
      Seek_thermal_pro_commands['Set factory settings features'],
      b'\x06\x00\x08\x00\x00\x00')
    self._write_data(Seek_thermal_pro_commands['Set firmware info features'],
                     b'\x17\x00')
    self._write_data(
      Seek_thermal_pro_commands['Set factory settings features'],
      b'\x01\x00\x00\x06\x00\x00')
    for i in range(10):
      for j in range(0, 256, 32):
        self._write_data(
          Seek_thermal_pro_commands['Set factory settings features'],
          b'\x20\x00' + bytes([j, i]) + b'\x00\x00')
    self._write_data(Seek_thermal_pro_commands['Set firmware info features'],
                     b'\x15\x00')
    self._write_data(Seek_thermal_pro_commands['Set image processing mode'],
                     b'\x08\x00')
    self._write_data(Seek_thermal_pro_commands['Set operation mode'],
                     b'\x01\x00')

    # Acquiring the dead pixels image and saving the dead pixes map
    self.log(logging.INFO, "Getting the dead pixels")
    for i in range(5):
      status, ret = self._grab()
      if status == 4:
        self._dead_pixels = self._get_dead_pixels_list(ret)
        break
      elif i == 4:
        self.log(logging.WARNING, "Could not get the dead pixels frame")

    # Acquiring the calibration image and calibrating the camera
    self.log(logging.INFO, "Calibrating the camera")
    for i in range(10):
      status, img = self._grab()
      if status == 1:
        self._calib = self._crop(img) - 1600
        break
      elif i == 9:
        raise TimeoutError("Could not set the camera")

  def get_image(self) -> tuple[float, np.ndarray]:
    """Reads a single image from the camera.

    Returns:
      The captured image as well as a timestamp.
    """

    count = 0
    # Looping until a valid frame is acquired
    while True:

      # Capturing one frame
      t = time()
      status, img = self._grab()

      # If a calibration frame is acquired, recalibrating
      if status == 1:
        self.log(logging.DEBUG, "Recalibrating the camera")
        self._calib = self._crop(img) - 1600

      # If a valid frame is acquired, returning it along with its metadata
      elif status == 3 and self._calib is not None:
        return t, self._correct_dead_pixels(self._crop(img)-self._calib)

      # If no valid image can be read, that's bad news
      elif count == 5:
        raise TimeoutError("Unable to read image")
      count += 1

  def close(self) -> None:
    """Resets the camera and releases the USB resources."""

    if self._dev is not None:
      for _ in range(3):
        self._write_data(Seek_thermal_pro_commands['Set operation mode'],
                         b'\x00\x00')
      self.log(logging.INFO, "Releasing the USB resources")
      usb.util.dispose_resources(self._dev)

  def _grab(self) -> [bytes, np.ndarray]:
    """Captures a raw image from the camera.

    Returns:
      The status information and the raw image.
    """

    # Sending the read command
    self._write_data(Seek_thermal_pro_commands['Start get image transfer'],
                     b'\x58\x5b\x01\x00')

    to_read = 2 * \
        Seek_thermal_pro_dimensions['Raw width'] * \
        Seek_thermal_pro_dimensions['Raw height']
    ret = bytearray()

    # Reading all the chunks containing the frame information
    while to_read - len(ret) > 512:
      ret += self._dev.read(
        endpoint=Seek_therm_usb_req['Read_img'],
        size_or_buffer=int(to_read / (Seek_thermal_pro_dimensions['Raw height']
                           / 20)),
        timeout=1000)

    # Returning the read image in the right format and the associated status
    status = ret[4]
    if len(ret) == to_read:
      return status, np.frombuffer(ret, dtype=np.uint16).reshape(
        Seek_thermal_pro_dimensions['Raw height'],
        Seek_thermal_pro_dimensions['Raw width'])
    else:
      return status, None

  def _get_dead_pixels_list(self, data: np.ndarray) -> list[tuple[Any]]:
    """Identifies the dead pixels on an image.

    Args:
      data: The image to identify dead pixels on.

    Returns:
      A :obj:`list` containing the indexes of the dead pixels.
    """

    img = self._crop(np.frombuffer(data, dtype=np.uint16).reshape(
      Seek_thermal_pro_dimensions['Raw height'],
      Seek_thermal_pro_dimensions['Raw width']))
    return list(zip(*np.where(img < 100)))

  @staticmethod
  def _crop(raw_img: np.ndarray) -> np.ndarray:
    """Simply crops an image to the right dimensions."""

    return raw_img[4: 4 + Seek_thermal_pro_dimensions['Height'],
                   1: 1 + Seek_thermal_pro_dimensions['Width']]

  def _correct_dead_pixels(self, img: np.ndarray) -> np.ndarray:
    """Corrects the dead pixels values.

    The new value is the average value of the surrounding pixels.

    Args:
      img: The image to correct.

    Returns:
      The corrected image.
    """

    for i, j in self._dead_pixels:
      img[i, j] = float(np.median(img[max(0, i - 1): i + 2,
                                      max(0, j - 1): j + 2]))
    return img

  def _write_data(self, request: int, data: bytes) -> int:
    """Wrapper for sending USB messages."""

    self.log(logging.DEBUG, f"Sending USB command with request type "
                            f"{Seek_therm_usb_req['Write']}, request "
                            f"{request}, value {0}, index {0},length or "
                            f"data {data}")

    try:
      return self._dev.ctrl_transfer(bmRequestType=Seek_therm_usb_req['Write'],
                                     bRequest=request,
                                     wValue=0,
                                     wIndex=0,
                                     data_or_wLength=data,
                                     timeout=None)
    except usb.core.USBError:
      raise IOError("An error occurred during USB communication")

  def _read_data(self, request: int, data: bytes) -> int:
    """Wrapper for reading USB messages."""

    self.log(logging.DEBUG, f"Sending USB command with request type "
                            f"{Seek_therm_usb_req['Read']}, request {request},"
                            f" value {0}, index {0},length or data {data}")

    try:
      return self._dev.ctrl_transfer(bmRequestType=Seek_therm_usb_req['Read'],
                                     bRequest=request,
                                     wValue=0,
                                     wIndex=0,
                                     data_or_wLength=data,
                                     timeout=None)
    except usb.core.USBError:
      raise IOError("An error occurred during USB communication")
