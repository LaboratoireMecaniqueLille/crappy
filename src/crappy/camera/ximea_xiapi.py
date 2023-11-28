# coding: utf-8

# TODO:
#  Add downsampling ( self.cam.set_downsampling('XI_DWN_2x2') )
#  Add region of interest selection
#  Add possibility to adjust the timeout

from time import time
from typing import Optional, Tuple
import numpy as np
import logging

from .meta_camera import Camera
from .._global import OptionalModule

try:
  from ximea import xiapi
except (ModuleNotFoundError, ImportError):
  xiapi = OptionalModule("ximea", "To use XiAPI cameras, please install the "
                         "official ximea Python module")


class XiAPI(Camera):
  """This class can read images from any of the Ximea cameras.

  It heavily relies on the :mod:`ximea` module, distributed by Ximea, which is
  itself a wrapper around the XiAPI low-level library.

  Note:
    Both the Python module and the camera drivers have to be installed from
    Ximea's website in order for this class to run.
  """

  def __init__(self) -> None:
    """Instantiates a Ximea Camera and a Ximea Image object."""

    super().__init__()

    self._cam = None
    self._started = False

    self._cam = xiapi.Camera()
    self._img = xiapi.Image()

    # Stores the last requested or read trigger mode value
    self._trig = 'Free run'

  def open(self, serial_number: Optional[str] = None, **kwargs) -> None:
    """Opens the connection to the camera, instantiates the available settings
    and starts the acquisition.

    Also sets custom values for the settings if provided by the user, otherwise
    sets them to their default.

    Args:
      serial_number: A :obj:`str` containing the serial number of the camera to
        open, in case several cameras are connected. If not provided and
        several cameras are available, one of them will be opened randomly.
      **kwargs: Values of the settings to set before opening the camera. Mostly
       useful if the configuration window is not used.
    """

    if serial_number is not None:
      self.log(logging.INFO, f"Opening the connection to the camera with "
                             f"serial number {serial_number}")
      self._cam.open_device_by_SN(serial_number)
    else:
      self.log(logging.INFO, "Opening the connection to the camera")
      self._cam.open_device()

    self.add_scale_setting('width', 1, self._get_w(), self._get_w, self._set_w,
                           self._get_w())
    self.add_scale_setting('height', 1, self._get_h(), self._get_h,
                           self._set_h, self._get_h())
    self.add_scale_setting('xoffset', 0, self._get_w(), self._get_ox,
                           self._set_ox, 0)
    self.add_scale_setting('yoffset', 0, self._get_h(), self._get_oy,
                           self._set_oy, 0)
    self.add_scale_setting('exposure', 28, 500000, self._get_exp,
                           self._set_exp, 10000)
    self.add_scale_setting('gain', 0., 6., self._get_gain, self._set_gain)
    self.add_bool_setting('AEAG', self._get_aeag, self._set_aeag, False)
    self.add_trigger_setting(self._get_extt, self._set_ext_trig)

    self.set_all(**kwargs)
    self.log(logging.INFO, "Starting the image acquisition")
    
    self._cam.start_acquisition()
    self._started = True

  def get_image(self) -> Tuple[float, np.ndarray]:
    """Reads a frame from the camera, and returns it along with its
    timestamp."""

    self._cam.get_image(self._img)
    return time(), self._img.get_image_data_numpy()

  def close(self) -> None:
    """Closes the connection to the camera and releases the resources."""

    if self._cam is not None:
      self.log(logging.INFO, "Closing the connection to the camera")
      self._cam.close_device()

  def _get_w(self) -> int:
    """Returns the width in pixels for selecting a region of interest."""

    return self._cam.get_width()

  def _get_h(self) -> int:
    """Returns the height in pixels for selecting a region of interest."""

    return self._cam.get_height()

  def _get_ox(self) -> int:
    """Returns the `x` offset in pixels for selecting a region of interest."""

    return self._cam.get_offsetX()

  def _get_oy(self) -> int:
    """Returns the `y` offset in pixels for selecting a region of interest."""

    return self._cam.get_offsetY()

  def _get_gain(self) -> float:
    """Returns the gain, in dB."""

    return self._cam.get_gain()

  def _get_exp(self) -> float:
    """Returns the exposure time, in microseconds."""

    return self._cam.get_exposure()

  def _get_aeag(self) -> bool:
    """Returns the auto exposure / auto gain parameter.

    It is either :obj:`True` or :obj:`False`.
    """

    return self._cam.get_param('aeag')

  def _get_extt(self) -> str:
    """Returns the current trigger mode value, and updates the last read
    trigger mode value if needed.

    The possible values for the trigger mode are `'Hardware'`, `'Free run'`,
    and `'Hdw after config'`.
    """

    r = self._cam.get_trigger_source()
    if r == 'XI_TRG_OFF' and self._trig == 'Hardware':
      self._trig = 'Free run'
    elif r != 'XI_TRG_OFF' and self._trig != 'Hardware':
      self._trig = 'Hardware'
    return self._trig

  def _set_w(self, width: int) -> None:
    """Sets the width in pixels for selecting a region of interest."""

    self._cam.set_width(width)

  def _set_h(self, height: int) -> None:
    """Sets the height in pixels for selecting a region of interest."""

    self._cam.set_height(height)

  def _set_ox(self, x_offset: int) -> None:
    """Sets the `x` offset in pixels for selecting a region of interest."""

    self._cam.set_offsetX(x_offset)

  def _set_oy(self, y_offset: int) -> None:
    """Sets the `y` offset in pixels for selecting a region of interest."""

    self._cam.set_offsetY(y_offset)

  def _set_gain(self, gain: float) -> None:
    """Sets the gain, in dB."""

    self._cam.set_gain(gain)

  def _set_exp(self, exposure: float) -> None:
    """Sets the exposure time, in microseconds."""

    self._cam.set_exposure(exposure)

  def _set_aeag(self, aeag: bool) -> None:
    """Sets the auto exposure / auto gain parameter.

    It is either :obj:`True` or :obj:`False`.
    """

    self._cam.set_param('aeag', int(aeag))

  def _set_ext_trig(self, trig: str) -> None:
    """Sets the requested trigger mode value, and updates the last requested
    trigger mode value.

    The possible values for the trigger mode are `'Hardware'`, `'Free run'`,
    and `'Hdw after config'`.
    """

    if self._started:
      self._cam.stop_acquisition()

    if trig == 'Hardware':
      self._cam.set_gpi_mode('XI_GPI_TRIGGER')
      self._cam.set_trigger_source('XI_TRG_EDGE_RISING')
    else:
      self._cam.set_gpi_mode('XI_GPI_OFF')
      self._cam.set_trigger_source('XI_TRG_OFF')
      
    self._trig = trig
    if self._started:
      self._cam.start_acquisition()
