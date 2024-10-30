# coding: utf-8

from time import time
from typing import Optional
import numpy as np
import logging
from  warnings import warn

from ..meta_camera import Camera
from ..._global import OptionalModule

try:
  from . import clModule as Cl
except (ImportError, ModuleNotFoundError):
  Cl = OptionalModule("clModule", "CameraLink module was not compiled. Please "
                                  "make sure /opt/SiliconSoftware/xxx/lib64 "
                                  "exists and reinstall Crappy")


class BaslerIronmanCameraLink(Camera):
  """This class can drive cameras over Camera Link through a Basler microEnable
  5 Ironman AD8 PoCL acquisition board.

  It is subclassed by the :class:`~crappy.lamcube.BiSpectral` and the
  :class:`~crappy.camera.cameralink.JaiGO5000CPMCL8Bits` Cameras. Not many
  settings can be accessed directly in Crappy, it is recommended to set them
  using a configuration file.

  Warning:
    This Camera relies on a custom-written C library that hasn't been tested in
    a long time. It might not be functional anymore. This Camera also requires
    proprietary drivers to be installed.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0
     renamed from *Cl_camera* to *BaslerIronmanCameraLink*
  .. versionremoved:: 2.1.0
  """

  def __init__(self) -> None:
    """Adds the frame rate setting."""

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be "
         f"deprecated and removed from Crappy. Please contact the maintainers "
         f"if you still use this Camera.", FutureWarning)

    self._cap = None

    super().__init__()

    self.add_scale_setting("framespersec", 1, 200, self._get_framespersec,
                           self._set_framespersec)

  def open(self,
           num_device: int = 0,
           config_file: Optional[str] = None,
           camera_type: Optional[str] = None,
           **kwargs) -> None:
    """Reads the settings from the arguments or from the configuration file,
    sets them on the camera, and starts the acquisition.

    Args:
      num_device: The index of the camera to open if multiple cameras are
        connected, as an :obj:`int`.

        .. versionchanged:: 2.0.0 renamed from *numdevice* to *num_device*
      config_file: Path to the configuration file for the camera, as a
        :obj:`str`. Allows setting various parameters at once, and to store
        them in a persistent way.
      camera_type: The type of camera to drive, as a :obj:`str`.
      **kwargs: All the settings to set on the camera.
    
    .. versionadded:: 1.5.10
       explicitly listing the *num_device*, *config_file* and *camera_type*
       arguments
    """

    # Reading the camera type from the config file, if applicable
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

    # Getting the data format from the kwargs
    if 'format' in kwargs:
      f = kwargs['format']
    # Getting the data format from the given camera type
    else:
      if camera_type[-1] == '8':
        f = Cl.FG_GRAY
      elif camera_type[-2:] == '16':
        f = Cl.FG_GRAY16
      elif camera_type[-2:] == '24':
        f = Cl.FG_COL24

      # Getting the data format from the configuration file
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

    # Opening the connection to the camera
    self.log(logging.INFO, "Initializing the communication with the camera")
    self._cap = Cl.VideoCapture()
    self._cap.open(num_device, camera_type, f)

    # Loading the parameters from the configuration file
    if config_file:
      self._cap.loadFile(config_file)

    self.set_all(**kwargs)

    # Starting the acquisition on the camera
    self.log(logging.INFO, "Starting acquisition")
    self._cap.startAcq()
    self._cap.set(Cl.FG_TRIGGERMODE, 1)
    self._cap.set(Cl.FG_EXSYNCON, 1)

  def get_image(self) -> tuple[float, np.ndarray]:
    """Reads a frame from the camera and returns it as is."""

    ret, frame = self._cap.read()
    t = time()
    if not ret:
      raise IOError("Could not read camera")

    return t, frame

  def close(self) -> None:
    """Stops the acquisition, and releases the resources attributed to the
    camera."""

    if self._cap is not None:
      self.log(logging.INFO, "Stopping acquisition")
      self._cap.stopAcq()
      self.log(logging.INFO, "Closing the communication with the camera")
      self._cap.release()

  def _set_framespersec(self, val: float) -> None:
    self._cap.set(Cl.FG_FRAMESPERSEC, val)

  def _get_framespersec(self) -> float:
    return self._cap.get(Cl.FG_FRAMESPERSEC)

  def _set_h(self, val: int) -> None:
    self._cap.stopAcq()
    self._cap.set(Cl.FG_HEIGHT, val)
    self._cap.startAcq()

  def _set_w(self, val: int) -> None:
    self._cap.stopAcq()
    self._cap.set(Cl.FG_WIDTH, val)
    self._cap.startAcq()

  def _get_h(self) -> int:
    return self._cap.get(Cl.FG_HEIGHT)

  def _get_w(self) -> int:
    return self._cap.get(Cl.FG_WIDTH)
