# coding: utf-8

import logging
from collections.abc import Callable
from typing import Any

from .camera_setting import CameraSetting


class CameraBoolSetting(CameraSetting):
  """Camera setting that can only be :obj:`True` or :obj:`False`.

  It is a child of
  :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting`.
  
  .. versionadded:: 1.5.10
  .. versionchanged:: 2.0.0
     renamed from *Camera_bool_setting* to *CameraBoolSetting*
  """

  def __init__(self,
               name: str,
               getter: Callable[[], bool] | None = None,
               setter: Callable[[bool], None] | None = None,
               default: bool = True) -> None:
    """Sets the attributes.

    Args:
      name: The name of the setting, that will be displayed in the GUI.
      getter: The method for getting the current value of the setting.
      setter: The method for setting the current value of the setting.
      default: The default value to assign to the setting.
    """

    super().__init__(name, getter, setter, default)

  @property
  def value(self) -> bool:
    """Returns the current value of the setting."""

    return super().value

  @value.setter
  def value(self, val: Any) -> None:
    if not isinstance(val, bool):
      raise TypeError(f"Only bool values are allowed for setting {self.name}")

    self.log(logging.DEBUG, f"Setting the setting {self.name} to {val}")
    self.was_set = True
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

    # Update the GUI, in case the value was modified via a reload() call
    if self.tk_var is not None:
      self.tk_var.set(self.value)
