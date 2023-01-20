# coding: utf-8

from typing import Callable, Optional, Union, Any
from multiprocessing import current_process
import logging

NbrType = Union[int, float]


class CameraSetting:
  """Base class for each camera setting.

  It is meant to be subclassed and should not be used as is.
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
    """"""

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)

  @property
  def value(self) -> Any:
    """Returns the current value of the setting, by calling the getter if one
    was provided or else by returning the stored value."""

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
      self.log(logging.WARNING, f"Could not set {self.name} to {val}, the "
                                f"value is {self.value} !")
