# coding: utf-8

import numpy as np
import logging
from warnings import warn

from ..camera.cameralink import BaslerIronmanCameraLink

table = (0x0000, 0xC0C1, 0xC181, 0x0140, 0xC301, 0x03C0, 0x0280, 0xC241,
         0xC601, 0x06C0, 0x0780, 0xC741, 0x0500, 0xC5C1, 0xC481, 0x0440,
         0xCC01, 0x0CC0, 0x0D80, 0xCD41, 0x0F00, 0xCFC1, 0xCE81, 0x0E40,
         0x0A00, 0xCAC1, 0xCB81, 0x0B40, 0xC901, 0x09C0, 0x0880, 0xC841,
         0xD801, 0x18C0, 0x1980, 0xD941, 0x1B00, 0xDBC1, 0xDA81, 0x1A40,
         0x1E00, 0xDEC1, 0xDF81, 0x1F40, 0xDD01, 0x1DC0, 0x1C80, 0xDC41,
         0x1400, 0xD4C1, 0xD581, 0x1540, 0xD701, 0x17C0, 0x1680, 0xD641,
         0xD201, 0x12C0, 0x1380, 0xD341, 0x1100, 0xD1C1, 0xD081, 0x1040,
         0xF001, 0x30C0, 0x3180, 0xF141, 0x3300, 0xF3C1, 0xF281, 0x3240,
         0x3600, 0xF6C1, 0xF781, 0x3740, 0xF501, 0x35C0, 0x3480, 0xF441,
         0x3C00, 0xFCC1, 0xFD81, 0x3D40, 0xFF01, 0x3FC0, 0x3E80, 0xFE41,
         0xFA01, 0x3AC0, 0x3B80, 0xFB41, 0x3900, 0xF9C1, 0xF881, 0x3840,
         0x2800, 0xE8C1, 0xE981, 0x2940, 0xEB01, 0x2BC0, 0x2A80, 0xEA41,
         0xEE01, 0x2EC0, 0x2F80, 0xEF41, 0x2D00, 0xEDC1, 0xEC81, 0x2C40,
         0xE401, 0x24C0, 0x2580, 0xE541, 0x2700, 0xE7C1, 0xE681, 0x2640,
         0x2200, 0xE2C1, 0xE381, 0x2340, 0xE101, 0x21C0, 0x2080, 0xE041,
         0xA001, 0x60C0, 0x6180, 0xA141, 0x6300, 0xA3C1, 0xA281, 0x6240,
         0x6600, 0xA6C1, 0xA781, 0x6740, 0xA501, 0x65C0, 0x6480, 0xA441,
         0x6C00, 0xACC1, 0xAD81, 0x6D40, 0xAF01, 0x6FC0, 0x6E80, 0xAE41,
         0xAA01, 0x6AC0, 0x6B80, 0xAB41, 0x6900, 0xA9C1, 0xA881, 0x6840,
         0x7800, 0xB8C1, 0xB981, 0x7940, 0xBB01, 0x7BC0, 0x7A80, 0xBA41,
         0xBE01, 0x7EC0, 0x7F80, 0xBF41, 0x7D00, 0xBDC1, 0xBC81, 0x7C40,
         0xB401, 0x74C0, 0x7580, 0xB541, 0x7700, 0xB7C1, 0xB681, 0x7640,
         0x7200, 0xB2C1, 0xB381, 0x7340, 0xB101, 0x71C0, 0x7080, 0xB041,
         0x5000, 0x90C1, 0x9181, 0x5140, 0x9301, 0x53C0, 0x5280, 0x9241,
         0x9601, 0x56C0, 0x5780, 0x9741, 0x5500, 0x95C1, 0x9481, 0x5440,
         0x9C01, 0x5CC0, 0x5D80, 0x9D41, 0x5F00, 0x9FC1, 0x9E81, 0x5E40,
         0x5A00, 0x9AC1, 0x9B81, 0x5B40, 0x9901, 0x59C0, 0x5880, 0x9841,
         0x8801, 0x48C0, 0x4980, 0x8941, 0x4B00, 0x8BC1, 0x8A81, 0x4A40,
         0x4E00, 0x8EC1, 0x8F81, 0x4F40, 0x8D01, 0x4DC0, 0x4C80, 0x8C41,
         0x4400, 0x84C1, 0x8581, 0x4540, 0x8701, 0x47C0, 0x4680, 0x8641,
         0x8201, 0x42C0, 0x4380, 0x8341, 0x4100, 0x81C1, 0x8081, 0x4040)


def calc_string(st: str, crc: int) -> int:
    """Given a binary string and starting CRC, Calc a final CRC-16."""

    for ch in st:
        crc = (crc >> 8) ^ table[(crc ^ ord(ch)) & 0xFF]
    return crc


def add_crc(s: str) -> str:
  """Wrapper for adding a CRC to a serial command."""

  return s + hex(calc_string(s, 0xFFFF)).split('x')[1].upper().rjust(4, '0')


def check_crc(s: str) -> bool:
  """Wrapper for checking whether a CRC is valid."""

  r = s[:-4]
  return add_crc(r) == s


def hexlify(n: int) -> str:
  """Converts an integer to its hexadecimal representation."""

  return hex(n).split('x')[1].rjust(2, '0').upper()


class BiSpectral(BaslerIronmanCameraLink):
  """This class allows driving a bi-chromatic infrared camera, through a Basler
  microEnable 5 Ironman AD8 PoCL acquisition board.

  It is a child of the
  :class:`~crappy.camera.cameralink.BaslerIronmanCameraLink` Camera. It can set
  various settings on the camera, including the ROI or the trigger mode.

  The bi-chromatic camera is a very specific setup, and won't certainly be ever
  used outside the LaMcube laboratory. This is why it is stored in the LaMcube
  submodule.

  Warning:
    This Camera relies on a custom-written C library that hasn't been tested in
    a long time. It might not be functional anymore. This Camera also requires
    proprietary drivers to be installed.
    
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Bispectral* to *BiSpectral*
  .. versionremoved:: 2.1.0
  """

  def __init__(self) -> None:
    """Adds the various setting for the Camera."""

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be "
         f"deprecated and removed from Crappy. Please contact the maintainers "
         f"if you still use this Camera.", FutureWarning)

    super().__init__()
    self.add_scale_setting('width', 1, 640, self._get_w, self._set_w, 640)
    self.add_scale_setting('height', 1, 512, self._get_h, self._set_h, 512)
    self.add_scale_setting('xoffset', 0, 1278, self._get_ox, self._set_ox, 0)
    self.add_scale_setting('yoffset', 0, 511, self._get_oy, self._set_oy, 0)
    self.add_scale_setting('IT1', 10, 10000, self._get_it1, self._set_it1)
    self.add_scale_setting('IT2', 10, 10000, self._get_it2, self._set_it2)
    self.add_scale_setting('fps', 1., 150., self._get_trigg_freq,
                           self._set_trigg_freq)
    self.add_trigger_setting(setter=self._set_external_trigger)

  def open(self,
           camera_type: str = 'SingleAreaGray2DShading',
           **kwargs) -> None:
    """Opens the Camera and sends initialization commands.
    
    .. versionadded:: 1.5.10 explicitly listing *camera_type* argument
    """

    super().open(camera_type=camera_type, **kwargs)

    self._send_cmd('@W1A084')  # Restore unwindowed Mode
    self._send_cmd('@W10012')  # Make sure the image is not inverted

  def get_image(self) -> tuple[float, np.ndarray]:
    """Grabs an image using the parent class' method, transforms it, and
    returns it."""

    t, frame = super().get_image()
    img = np.ones((self.height, self.width * 2), dtype=np.uint8)
    img[::, :self.width:2] = frame[::, ::4]
    img[::, 1:self.width:2] = frame[::, 1::4]
    img[::, self.width::2] = frame[::, 2::4]
    img[::, self.width + 1::2] = frame[::, 3::4]
    return t, img

  def _send_cmd(self, cmd: str) -> str:
    """Wrapper for sending a command to the Camera."""

    self.log(logging.DEBUG, f"Sending command {cmd}")
    r = self._cap.serialWrite(add_crc(cmd))
    if not check_crc(r) or r[1] != 'Y':
      self.log(logging.WARNING, f"Incorrect reply {r}")
    return r[2:4]

  def _set_external_trigger(self, val: str) -> None:
    """Sets the external trigger to val by toggling the value of the 3rd bit
    of register 102."""

    if val == 'Hardware':
      self._send_cmd('@W1027C')  # 3rd bit to 1
    else:
      self._send_cmd('@W10274')  # 3rd bit to 0

  def _get_roi(self) -> tuple[int, int, int, int]:
    """Returns the minimum and maximum x and y coordinates of the current
    ROI."""

    x1min_lsb = self._send_cmd("@R1D0")
    x1min_msb = self._send_cmd("@R1D1")
    y1min_lsb = self._send_cmd("@R1D2")
    y1min_msb = self._send_cmd("@R1D3")
    x1max_lsb = self._send_cmd("@R1D4")
    x1max_msb = self._send_cmd("@R1D5")
    y1max_lsb = self._send_cmd("@R1D6")
    y1max_msb = self._send_cmd("@R1D7")
    xmin = int(x1min_msb + x1min_lsb, 16)
    xmax = int(x1max_msb + x1max_lsb, 16)
    ymin = int(y1min_msb + y1min_lsb, 16)
    ymax = int(y1max_msb + y1max_lsb, 16)
    return xmin, ymin, xmax, ymax

  def _set_roi(self, xmin: int, ymin: int, xmax: int, ymax: int) -> None:
    """Sets the minimum and maximum x and y coordinates of the ROI."""

    if (xmin, xmax, ymin, ymax) != (0, 0, 639, 511):
      self._send_cmd('@W1A080')  # Set to windowed mode
    else:
      self._send_cmd('@W1A084')
      self.log(logging.INFO, f"D set ROI to {xmin}, {ymin}, {xmax}, {ymax}")
    lsb_xmin = hexlify(xmin % 256)
    msb_xmin = hexlify(xmin // 256)
    lsb_xmax = hexlify(xmax % 256)
    msb_xmax = hexlify(xmax // 256)
    lsb_ymin = hexlify(ymin % 256)
    msb_ymin = hexlify(ymin // 256)
    lsb_ymax = hexlify(ymax % 256)
    msb_ymax = hexlify(ymax // 256)
    self._send_cmd("@W1D0" + lsb_xmin)
    self._send_cmd("@W1D1" + msb_xmin)
    self._send_cmd("@W1D2" + lsb_ymin)
    self._send_cmd("@W1D3" + msb_ymin)
    self._send_cmd("@W1D4" + lsb_xmax)
    self._send_cmd("@W1D5" + msb_xmax)
    self._send_cmd("@W1D6" + lsb_ymax)
    self._send_cmd("@W1D7" + msb_ymax)

  def _get_it(self) -> tuple[float, float]:
    """Reads the integration time from the Camera and returns it."""

    mc = 10.35  # MHz
    it1_lsb = self._send_cmd("@R1B4")
    it1_mid = self._send_cmd("@R1B5")
    it1_msb = self._send_cmd("@R1B6")
    it2_lsb = self._send_cmd("@R1B8")
    it2_mid = self._send_cmd("@R1B9")
    it2_msb = self._send_cmd("@R1BA")
    it1 = int(it1_msb + it1_mid + it1_lsb, 16)  # Number of clock cycles
    it2 = int(it2_msb + it2_mid + it2_lsb, 16)
    return it1 / mc, it2 / mc  # IT in µs

  def _set_it(self, it1: int, it2: int) -> None:
    """Sets the integration time on the Camera."""

    mc = 10.35
    it1 = int(mc * it1)
    it2 = int(mc * it2)
    it1_lsb = hexlify(it1 % 256)
    it1 -= it1 % 256
    it1 //= 256
    it1_mid = hexlify(it1 % 256)
    it1_msb = hexlify(it1 // 256)
    it2_lsb = hexlify(it2 % 256)
    it2 -= it2 % 256
    it2 //= 256
    it2_mid = hexlify(it2 % 256)
    it2_msb = hexlify(it2 // 256)
    self._send_cmd("@W1B4" + it1_lsb)
    self._send_cmd("@W1B5" + it1_mid)
    self._send_cmd("@W1B6" + it1_msb)
    self._send_cmd("@W1B8" + it2_lsb)
    self._send_cmd("@W1B9" + it2_mid)
    self._send_cmd("@W1BA" + it2_msb)

  def _get_trigg_freq(self) -> float:
    """Reads the trigger frequency from the Camera."""

    mc = 10350000  # Hz
    p_lsb = self._send_cmd("@R1B0")
    p_mid = self._send_cmd("@R1B1")
    p_msb = self._send_cmd("@R1B2")
    p = int(p_msb + p_mid + p_lsb, 16)
    return mc / p

  def _set_trigg_freq(self, freq: float) -> None:
    """Sets the trigger frequency on the Camera."""

    mc = 10350000  # Hz
    period = int(mc / freq)
    p_lsb = hexlify(period % 256)
    period -= period % 256
    period //= 256
    p_mid = hexlify(period % 256)
    p_msb = hexlify(period // 256)
    self._send_cmd("@W1B0" + p_lsb)
    self._send_cmd("@W1B1" + p_mid)
    self._send_cmd("@W1B2" + p_msb)

  def _set_w(self, val: int) -> None:
    super()._set_w(val * 2)
    self._set_roi(self.xoffset, self.yoffset, self.xoffset + self.width - 1,
                  self.yoffset + self.height - 1)

  def _get_w(self) -> int:
    return int(super()._get_w() / 2)

  def _set_h(self, val: int) -> None:
    super()._set_h(val)
    self._set_roi(self.xoffset, self.yoffset, self.xoffset + self.width - 1,
                  self.yoffset + self.height - 1)

  def _set_ox(self, val: int) -> None:
    self._set_roi(val, self.yoffset, val + self.width - 1,
                  self.yoffset + self.height - 1)

  def _set_oy(self, val: int) -> None:
    self._set_roi(self.xoffset, val, self.xoffset + self.width - 1,
                  val + self.height - 1)

  def _get_ox(self) -> int:
    return self._get_roi()[0]

  def _get_oy(self) -> int:
    return self._get_roi()[1]

  def _get_it1(self) -> int:
    return int(self.get_itT()[0])

  def _get_it2(self) -> int:
    return int(self._get_it()[1])

  def _set_it1(self, val: int) -> None:
    self._set_it(val, self._get_it2())

  def _set_it2(self, val: int) -> None:
    self._set_it(self._get_it1(), val)
