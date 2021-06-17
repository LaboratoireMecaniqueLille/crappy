# coding: utf-8

import time

from .camera import Camera
from .._global import OptionalModule
try:
  from ximea import xiapi
except (ModuleNotFoundError, ImportError):
  xiapi = OptionalModule("ximea", "To use Xiapi cameras, please install the "
      "official ximea Python module")


class Xiapi(Camera):
  """Camera class for ximeas using official XiAPI."""

  def __init__(self):
    Camera.__init__(self)
    self.name = "Xiapi"
    self.cam = xiapi.Camera()
    self.img = xiapi.Image()
    self.add_setting("width", self._get_w, self._set_w, (1, self._get_w))
    self.add_setting("height", self._get_h, self._set_h, (1, self._get_h))
    self.add_setting("xoffset", self._get_ox, self._set_ox, (0, self._get_w))
    self.add_setting("yoffset", self._get_oy, self._set_oy, (0, self._get_h))
    self.add_setting("exposure", self._get_exp, self._set_exp,
                     (28, 500000), 10000)
    self.add_setting("gain", self._get_gain, self._set_gain, (0., 6.))
    self.add_setting("AEAG", self._get_aeag, self._set_aeag, True, False)
    self.add_setting("external_trig", self._get_extt, self._set_extt,
                     True, False)

  def _get_w(self):
    return self.cam.get_width()

  def _get_h(self):
    return self.cam.get_height()

  def _get_ox(self):
    return self.cam.get_offsetX()

  def _get_oy(self):
    return self.cam.get_offsetY()

  def _get_gain(self):
    return self.cam.get_gain()

  def _get_exp(self):
    return self.cam.get_exposure()

  def _get_aeag(self):
    return self.cam.get_param('aeag')

  def _get_extt(self):
    r = self.cam.get_trigger_source()
    if r == 'XI_TRG_OFF':
      return False
    else:
      return True

  def _set_w(self, width):
    self.cam.set_width(width)

  def _set_h(self, height):
    self.cam.set_height(height)

  def _set_ox(self, x_offset):
    self.cam.set_offsetX(x_offset)

  def _set_oy(self, y_offset):
    self.cam.set_offsetY(y_offset)

  def _set_gain(self, gain):
    self.cam.set_gain(gain)

  def _set_exp(self, exposure):
    self.cam.set_exposure(exposure)

  def _set_aeag(self, aeag):
    self.cam.set_param('aeag', int(aeag))

  def _set_extt(self, trig):
    if trig:
      self.cam.set_gpi_mode('XI_GPI_TRIGGER')
      self.cam.set_trigger_source('XI_TRG_EDGE_RISING')
    else:
      self.cam.set_gpi_mode('XI_GPI_OFF')
      self.cam.set_trigger_source('XI_TRG_OFF')

  def open(self, sn=None, **kwargs):
    """Will actually open the camera, args will be set to default unless
    specified otherwise in kwargs.

    If `sn` is given, it will open the camera with the corresponding serial
    number.

    Else, it will open any camera.
    """

    self.sn = sn
    # self.close()
    if self.sn is not None:
      self.cam.open_device_by_SN(self.sn)
    else:
      self.cam.open_device()

    for k in kwargs:
      assert k in self.available_settings, str(self) +\
                                           "Unexpected kwarg: " + str(k)
    self.set_all(**kwargs)
    self.set_all(**kwargs)
    self.cam.start_acquisition()

  def reopen(self, **kwargs):
    """Will reopen the camera, args will be set to default unless specified
    otherwise in kwargs.
    """

    self.open()
    self.set_all(override=True, **kwargs)

  def get_image(self):
    """This method get a frame on the selected camera and return a ndarray.

    Returns:
        frame from ximea device (`ndarray height*width`).
    """

    self.cam.get_image(self.img)
    t = time.time()
    return t, self.img.get_image_data_numpy()

  def close(self):
    """This method closes properly the camera.

    Returns:
        void return function.
    """

    self.cam.close_device()
