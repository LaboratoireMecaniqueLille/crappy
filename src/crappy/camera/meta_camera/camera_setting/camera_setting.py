# coding: utf-8

from typing import Optional, Union, Any
from collections.abc import Callable
from multiprocessing import current_process
import logging

NbrType = Union[int, float]


class CameraSetting:
  """Base class for each Camera setting.

  It is meant to be subclassed and should not be used as is.

  The Camera setting classes hold all the information needed to read and set a
  setting of a :class:`~crappy.camera.Camera` object. Several types of settings
  are defined, as children of this class :
  :class:`~crappy.camera.meta_camera.camera_setting.CameraBoolSetting`,
  :class:`~crappy.camera.meta_camera.camera_setting.CameraChoiceSetting`,
  and :class:`~crappy.camera.meta_camera.camera_setting.CameraScaleSetting`.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Camera_setting* to *CameraSetting*
  """

  def __init__(self,
               name: str,
               getter: Callable[[], Any],
               setter: Callable[[Any], None],
               default: Any) -> None:
    """Sets the attributes.

    Args:
      name: The name of the setting, that will be displayed in the GUI.
      getter: The method for getting the current value of the setting.
      setter: The method for setting the current value of the setting.
      default: The default value to assign to the setting.
    """

    # Attributes shared by all the settings
    self.name = name
    self.default = default
    self.type = type(default)

    # Attributes used in the GUI
    self.tk_var = None
    self.tk_obj = None

    # Attributes for internal use only
    self._value_no_getter = default
    self._getter = getter
    self._setter = setter
    self._logger: Optional[logging.Logger] = None

  def log(self, level: int, msg: str) -> None:
    """Records log messages for the CameraSetting.

    Also instantiates the logger when logging the first message.

    Args:
      level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.

    .. versionadded:: 2.0.0
    """

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)

  @property
  def value(self) -> Any:
    """Returns the current value of the setting, by calling the getter if one
    was provided or else by returning the stored value.

    When the getter is called, calls the setter if one was provided and updates
    the sored value. After calling the setter, checks that the value was set
    by calling the getter and displays a warning message if the target and
    actual values don't match.
    """

    if self._getter is not None:
      return self._getter()
    else:
      return self._value_no_getter

  @value.setter
  def value(self, val: Any) -> None:
    self.log(logging.DEBUG, f"Setting the setting {self.name} to {val}")
    self._value_no_getter = val
    if self._setter is not None:
      self._setter(val)

    if self.value != val:
      # Double-checking, got strange behavior sometimes probably because of
      # delays in lower level APIs
      if self.value == val:
        return
      self.log(logging.WARNING, f"Could not set {self.name} to {val}, the "
                                f"value is {self.value} !")

  def reload(self, *_, **__) -> None:
    """Allows modifying a setting once it is already being displayed in the
    GUI.

    Mostly helpful for adjusting the ranges of sliders.

    .. versionadded:: 2.0.0
    """

    ...
