# coding: utf-8

# Todo:
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
  """Camera class for ximeas using official XiAPI."""

  def __init__(self) -> None:
    """"""

    super().__init__()

    self._cam = None

    self._cam = xiapi.Camera()
    self._img = xiapi.Image()

    # Stores the last requested or read trigger mode value
    self._trig = 'Free run'

  def open(self, sn: Optional[str] = None, **kwargs) -> None:
    """Will actually open the camera, args will be set to default unless
    specified otherwise in kwargs.

    If `sn` is given, it will open the camera with the corresponding serial
    number.

    Else, it will open any camera.
    """

    if sn is not None:
      self.log(logging.INFO, f"Opening the connection to the camera with "
                             f"serial number {sn}")
      self._cam.open_device_by_SN(sn)
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
    self.add_trigger_setting(self._get_extt, self._set_extt)

    self.set_all(**kwargs)
    self.log(logging.INFO, "Starting the image acquisition")
    self._cam.start_acquisition()

  def get_image(self) -> Tuple[float, np.ndarray]:
    """This method get a frame on the selected camera and return a ndarray.

    Returns:
        frame from ximea device (`ndarray height*width`).
    """

    self._cam.get_image(self._img)
    t = time()
    self._frame_nr += 1
    return t, self._img.get_image_data_numpy()

  def close(self) -> None:
    """This method closes properly the camera.

    Returns:
        void return function.
    """

    if self._cam is not None:
      self.log(logging.INFO, "Closing the connection to the camera")
      self._cam.close_device()

  def _get_w(self) -> int:
    return self._cam.get_width()

  def _get_h(self) -> int:
    return self._cam.get_height()

  def _get_ox(self) -> int:
    return self._cam.get_offsetX()

  def _get_oy(self) -> int:
    return self._cam.get_offsetY()

  def _get_gain(self) -> float:
    return self._cam.get_gain()

  def _get_exp(self) -> float:
    return self._cam.get_exposure()

  def _get_aeag(self) -> bool:
    return self._cam.get_param('aeag')

  def _get_extt(self) -> str:
    """Returns the current trigger mode value, and updates the last read
    trigger mode value if needed."""

    r = self._cam.get_trigger_source()
    if r == 'XI_TRG_OFF' and self._trig == 'Hardware':
      self._trig = 'Free run'
    elif r != 'XI_TRG_OFF' and self._trig != 'Hardware':
      self._trig = 'Hardware'
    return self._trig

  def _set_w(self, width: int) -> None:
    self._cam.set_width(width)

  def _set_h(self, height: int) -> None:
    self._cam.set_height(height)

  def _set_ox(self, x_offset: int) -> None:
    self._cam.set_offsetX(x_offset)

  def _set_oy(self, y_offset: int) -> None:
    self._cam.set_offsetY(y_offset)

  def _set_gain(self, gain: float) -> None:
    self._cam.set_gain(gain)

  def _set_exp(self, exposure: float) -> None:
    self._cam.set_exposure(exposure)

  def _set_aeag(self, aeag: bool) -> None:
    self._cam.set_param('aeag', int(aeag))

  def _set_extt(self, trig: str) -> None:
    """Sets the requested trigger mode value, and updates the last requested
    trigger mode value."""

    if trig == 'Hardware':
      self._cam.set_gpi_mode('XI_GPI_TRIGGER')
      self._cam.set_trigger_source('XI_TRG_EDGE_RISING')
    else:
      self._cam.set_gpi_mode('XI_GPI_OFF')
      self._cam.set_trigger_source('XI_TRG_OFF')
    self._trig = trig
