# coding: utf-8

from typing import Any
from collections.abc import Callable
from multiprocessing import current_process
import logging

NbrType = int | float


class CameraSetting:
  """Base class for each Camera setting.

  It is meant to be subclassed and should not be used as is.

  The Camera setting classes hold all the information needed to read and set a
  setting of a :class:`~crappy.camera.meta_camera.camera.Camera` object.
  Several types of settings are defined, as children of this class :
  :class:`~crappy.camera.meta_camera.camera_setting.CameraBoolSetting`,
  :class:`~crappy.camera.meta_camera.camera_setting.CameraChoiceSetting`,
  and :class:`~crappy.camera.meta_camera.camera_setting.CameraScaleSetting`.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Camera_setting* to *CameraSetting*
  """

  def __init__(self,
               name: str,
               getter: Callable[[], Any] | None,
               setter: Callable[[Any], None] | None,
               default: Any) -> None:
    """Sets the attributes.

    Args:
      name: The name of the setting, that will be displayed in the GUI.
      getter: The method for getting the current value of the setting.
      setter: The method for setting the current value of the setting.
      default: The default value to assign to the setting.
    """

    # Attributes shared by all the settings
    self.name: str = name
    self.default = default
    self.type = type(default)
    self.was_set: bool = False
    self.user_set: bool = False
    self._revision: int = 0
    self._reload_override_allowed: bool = False

    # Attributes for internal use only
    self._value_no_getter = default
    self._getter = getter
    self._setter = setter

    if not hasattr(self, '_logger'):
      self._logger: logging.Logger | None = None

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
  def revision(self) -> int:
    """A counter changed whenever the value or metadata is updated.

    Configuration interfaces can use it to synchronize their own controls
    without storing any GUI objects on this setting.
    """

    return self._revision

  def allow_reload_override(self) -> None:
    """Allow later reloads to replace values initially supplied by a user.

    Explicit camera kwargs are protected during camera setup. Once interactive
    configuration begins, those values become editable and can be replaced by
    dependent setting reloads.
    """

    self._reload_override_allowed = True

  def _mark_changed(self) -> None:
    """Record a value or metadata change for interested interfaces."""

    self._revision += 1

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
    self.was_set = True
    self._value_no_getter = val
    self._mark_changed()
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
    """Allows modifying a setting after it has been instantiated.

    Mostly helpful for adjusting the ranges of sliders.

    .. versionadded:: 2.0.0
    """

    ...
