# coding: utf-8

from typing import Optional, Callable, Union
import logging

from .camera_setting import CameraSetting

NbrType = Union[int, float]


class CameraScaleSetting(CameraSetting):
  """Camera setting that can take any value between a lower and an upper
  boundary.

  This class can handle settings that should only take :obj:`int` values as
  well as settings that can take :obj:`float` value.
  """

  def __init__(self,
               name: str,
               lowest: NbrType,
               highest: NbrType,
               getter: Optional[Callable[[], NbrType]] = None,
               setter: Optional[Callable[[NbrType], None]] = None,
               default: Optional[NbrType] = None) -> None:
    """Sets the attributes.

    Args:
      name: The name of the setting, that will be displayed in the GUI.
      lowest: The lower boundary for the setting values.
      highest: The upper boundary for the setting values.
      getter: The method for getting the current value of the setting.
      setter: The method for setting the current value of the setting.
      default: The default value to assign to the setting.
    """

    self.lowest = lowest
    self.highest = highest
    self.type = int if isinstance(lowest + highest, int) else float

    if default is None:
      default = self.type((lowest + highest) / 2)

    super().__init__(name, getter, setter, default)

  @property
  def value(self) -> NbrType:
    """Returns the current value of the setting, by calling the getter if one
    was provided or else by returning the stored value."""

    if self._getter is not None:
      return self.type(min(max(self._getter(), self.lowest), self.highest))
    else:
      return self.type(self._value_no_getter)

  @value.setter
  def value(self, val: NbrType) -> None:
    val = min(max(val, self.lowest), self.highest)
    self.log(logging.DEBUG, f"Setting the setting {self.name} to {val}")

    self._value_no_getter = self.type(val)
    if self._setter is not None:
      self._setter(self.type(val))

    if self.value != val:
      self.log(logging.WARNING, f"Could not set {self.name} to {val}, the "
                                f"value is {self.value} !")
