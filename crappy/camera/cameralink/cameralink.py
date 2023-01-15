# coding: utf-8

# Todo:
#  Update with recent Python bindings

from time import time
from typing import Optional, Tuple
import numpy as np
import logging

from ..meta_camera import Camera
from ..._global import OptionalModule
try:
  from . import clModule as Cl
except (ImportError, ModuleNotFoundError):
  Cl = OptionalModule("clModule", "CameraLink module was not compiled. Please "
                                  "make sure /opt/SiliconSoftware/xxx/lib64 "
                                  "exists and reinstall Crappy")


class Cl_camera(Camera):
  """Cameralink camera sensor."""

  def __init__(self) -> None:
    """Using the clModule, will open a cameraLink camera.

    Note:
      If a config file is specified, it will be used to configure the camera.

      If not set, it will be asked, unless set to :obj:`False` (or 0).

      Else, you must at least provide the camera type (eg: `"FullAreaGray8"`).

      Using a config file is recommended over changing all settings manually.
    """

    self._cap = None

    super().__init__()

    self.add_scale_setting("framespersec", 1, 200, self._get_framespersec,
                           self._set_framespersec)

  def open(self,
           numdevice: int = 0,
           config_file: Optional[str] = None,
           camera_type: Optional[str] = None,
           **kwargs) -> None:
    """Opens the camera."""

    if camera_type is None and config_file is not None:
      self.log(logging.INFO, "Reading config file for getting the type of "
                             "camera")
      with open(config_file, 'r') as file:
        r = file.readlines()
      r = [s for s in r if s[:5] == "Typ='"]
      if r:
        camera_type = r[0][5:-3]

    if camera_type is None:
      raise AttributeError("No camera type or valid config file specified!")

    if 'format' in kwargs:
      f = kwargs['format']

    else:
      if camera_type[-1] == '8':
        f = Cl.FG_GRAY
      elif camera_type[-2:] == '16':
        f = Cl.FG_GRAY16
      elif camera_type[-2:] == '24':
        f = Cl.FG_COL24

      else:
        if config_file:
          self.log(logging.WARNING, "Reading config file for getting the data "
                                    "format")
          with open(config_file, 'r') as file:
            r = file.readlines()
          r = [s for s in r if s[:10] == "FG_FORMAT="]
          if len(r) != 0:
            f = int(r[0].split('=')[1])
          else:
            raise ValueError("Could not determine the format")
        else:
          raise ValueError("Could not determine the format")

    self.log(logging.INFO, "Initializing the communication with the camera")
    self.cap = Cl.VideoCapture()
    self.cap.open(numdevice, camera_type, f)

    if config_file:
      self.cap.loadFile(config_file)

    self.set_all(**kwargs)

    self.log(logging.INFO, "Starting acquisition")
    self.cap.startAcq()
    self.cap.set(Cl.FG_TRIGGERMODE, 1)
    self.cap.set(Cl.FG_EXSYNCON, 1)

  def get_image(self) -> Tuple[float, np.ndarray]:
    """"""

    ret, frame = self.cap.read()
    t = time()
    if not ret:
      raise IOError("Could not read camera")

    return t, frame

  def close(self) -> None:
    """"""

    if self._cap is not None:
      self.log(logging.INFO, "Stopping acquisition")
      self.cap.stopAcq()
      self.log(logging.INFO, "Closing the communication with the camera")
      self.cap.release()

  def _set_framespersec(self, val: float) -> None:
    self.cap.set(Cl.FG_FRAMESPERSEC, val)

  def _get_framespersec(self) -> float:
    return self.cap.get(Cl.FG_FRAMESPERSEC)

  def _set_h(self, val: int) -> None:
    self.cap.stopAcq()
    self.cap.set(Cl.FG_HEIGHT, val)
    self.cap.startAcq()

  def _set_w(self, val: int) -> None:
    self.cap.stopAcq()
    self.cap.set(Cl.FG_WIDTH, val)
    self.cap.startAcq()

  def _get_h(self) -> int:
    return self.cap.get(Cl.FG_HEIGHT)

  def _get_w(self):
    return self.cap.get(Cl.FG_WIDTH)
