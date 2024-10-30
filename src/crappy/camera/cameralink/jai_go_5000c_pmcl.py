# coding: utf-8

from typing import Optional
import numpy as np
import logging
from  warnings import warn

from . import BaslerIronmanCameraLink
from ..._global import OptionalModule

try:
  from . import clModule as Cl
except (ModuleNotFoundError, ImportError):
  Cl = OptionalModule("clModule")

format_to_num = {'8 bits': 0,
                 '10 bits': 1,
                 '12 bits': 2}
num_to_format = {val: key for key, val in format_to_num.items()}


class JaiGO5000CPMCL8Bits(BaslerIronmanCameraLink):
  """This class can drive a JAI GO-5000C-PMCL camera in 8 bits mode, through a
  Basler microEnable 5 Ironman AD8 PoCL acquisition board.

  It is a child of the
  :class:`~crappy.camera.cameralink.BaslerIronmanCameraLink` Camera. The only
  difference is that it can only run in 8 bits mode, and gives access to more
  camera settings in Crappy. It is subclassed by the
  :class:`~crappy.camera.cameralink.JaiGO5000CPMCL` Camera, that allows driving
  the same hardware but in 10 and 12 bits mode.

  Warning:
    This Camera relies on a custom-written C library that hasn't been tested in
    a long time. It might not be functional anymore. This Camera also requires
    proprietary drivers to be installed.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Jai8* to *JaiGO5000CPMCL8Bits*
  .. versionremoved:: 2.1.0
  """

  def __init__(self) -> None:
    """Adds various settings to the Camera."""

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be "
         f"deprecated and removed from Crappy. Please contact the maintainers "
         f"if you still use this Camera.", FutureWarning)

    super().__init__()

    self.add_scale_setting('width', 1, 2560, self._get_w, self._set_w, 2560)
    self.add_scale_setting('height', 1, 2048, self._get_h, self._set_h, 2048)
    self.add_scale_setting('exposure', 10, 800000, self._get_exp,
                           self._set_exp)

  def open(self,
           config_file: Optional[str] = None,
           camera_type: str = 'FullAreaGray8',
           **kwargs) -> None:
    """Opens the connection to the camera using the parent class' method, and
    sets the image format to 8 bits.

    Args:
      config_file: Path to the configuration file for the camera, as a
        :obj:`str`. Allows setting various parameters at once, and to store
        them in a persistent way.
      camera_type: The type of camera to drive, as a :obj:`str`.
      **kwargs: All the settings to set on the camera.

    .. versionadded:: 1.5.10
       explicitly listing the *config_file* and *camera_type* arguments
    """

    super().open(config_file=config_file,
                 camera_type=camera_type,
                 **kwargs)

    self.log(logging.DEBUG, "Writing b'TAGM=5\\r\\n' to the camera")
    self._cap.serialWrite('TAGM=5\r\n')  # (default)
    self._set_format('8 bits')  # Set camera to 8 bits
    self._cap.set(Cl.FG_CAMERA_LINK_CAMTYP, 208)  # Set the input to 8 bits
    self._cap.set(Cl.FG_SENSORREADOUT, 0)  # Sets the correct framegrabber mode

  def _set_w(self, val: int) -> None:
    self.stopAcq()
    super()._set_w(val)
    self.log(logging.DEBUG, f"Writing b'WTC={val}\\r\\n' to the camera")
    self._cap.serialWrite('WTC={}\r\n'.format(val))
    self.startAcq()

  def _set_h(self, val: int) -> None:
    self.stopAcq()
    super()._set_h(val)
    self.log(logging.DEBUG, f"Writing b'HTL={val}\\r\\n' to the camera")
    self._cap.serialWrite('HTL={}\r\n'.format(val))
    self.startAcq()

  def _get_format(self) -> str:
    self.log(logging.DEBUG, "Writing b'BA?\\r\\n' to the camera")
    r = self._cap.serialWrite('BA?\r\n')
    return num_to_format[int(r[3])]

  def _set_format(self, val: str) -> None:
    self.log(logging.DEBUG, f"Writing b'BA={format_to_num[val]}\\r\\n' "
                            f"to the camera")
    self._cap.serialWrite('BA={}\r\n'.format(format_to_num[val]))

  def _set_exp(self, val: int) -> None:
    self.log(logging.DEBUG, f"Writing b'RE={val}\\r\\n' to the camera")
    self._cap.serialWrite('PE={}\r\n'.format(val))

  def _get_exp(self) -> int:
    self.log(logging.DEBUG, "Writing b'PE?\\r\\n' to the camera")
    return int(self._cap.serialWrite('PE?\r\n').strip()[3:])


class JaiGO5000CPMCL(JaiGO5000CPMCL8Bits):
  """This class can drive a JAI GO-5000C-PMCL camera in 10 or 12 bits mode,
  through a Basler microEnable 5 Ironman AD8 PoCL acquisition board.

  It is a child of the :class:`~crappy.camera.cameralink.JaiGO5000CPMCL8Bits`
  Camera. The only difference with its parent class is that it sets the 10 or
  12 bits mode on the camera, and modifies the acquired image before returning
  it.
  
  Warning:
    This Camera relies on a custom-written C library that hasn't been tested in
    a long time. It might not be functional anymore. This Camera also requires
    proprietary drivers to be installed.

  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Jai* to *JaiGO5000CPMCL*
  .. versionremoved:: 2.1.0
  """

  def __init__(self) -> None:
    """Adds the data_format settings to the Camera."""

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be "
         f"deprecated and removed from Crappy. Please contact the maintainers "
         f"if you still use this Camera.", FutureWarning)

    super().__init__()
    self.add_choice_setting('data_format', ('10 bits', '12 bits'),
                            self._get_format, self._set_format, '12 bits')

  def open(self,
           camera_type: str = 'MediumAreaGray16',
           **kwargs) -> None:
    """Opens the connection to the camera using the parent class' method, and
    sets the image format to 12 bits.

    Args:
      camera_type: The type of camera to drive, as a :obj:`str`.
      **kwargs: All the settings to set on the camera.

    .. versionadded:: 1.5.10 explicitly listing the *camera_type* argument
    """

    super().open(camera_type=camera_type, **kwargs)
    # dual tap (default does not allow 12 bits)
    self.log(logging.DEBUG, "Writing b'TAGM=1\\r\\n' to the camera")
    self._cap.serialWrite('TAGM=1\r\n')
    self._set_format('12 bits')  # 12 bits
    self._cap.set(Cl.FG_CAMERA_LINK_CAMTYP, 212)  # Set the input to 12 bits
    self._cap.set(Cl.FG_SENSORREADOUT, 7)  # Sets the correct framegrabber mode

  def get_image(self) -> tuple[float, np.ndarray]:
    """Grabs a frame using the parent class' method, and returns if after
    shifting bits."""

    t, frame = super().get_image()
    return t, frame >> 4
