# coding: utf-8

from typing import Optional, Union
from collections.abc import Callable
import logging

from .camera_setting import CameraSetting

NbrType = Union[int, float]


class CameraScaleSetting(CameraSetting):
  """Camera setting that can take any value between a lower and an upper
  boundary.

  It is a child of
  :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting`.

  This class can handle settings that should only take :obj:`int` values as
  well as settings that can take :obj:`float` value. The type used is
  :obj:`int` is both of the given lowest or highest values are :obj:`int`,
  otherwise :obj:`float` is used.
  
  .. versionadded:: 1.5.10
  .. versionchanged:: 2.0.0
     renamed from *Camera_scale_setting* to *CameraScaleSetting*
  """

  def __init__(self,
               name: str,
               lowest: NbrType,
               highest: NbrType,
               getter: Optional[Callable[[], NbrType]] = None,
               setter: Optional[Callable[[NbrType], None]] = None,
               default: Optional[NbrType] = None,
               step: Optional[NbrType] = None) -> None:
    """Sets the attributes.

    Args:
      name: The name of the setting, that will be displayed in the GUI.
      lowest: The lower boundary for the setting values.
      highest: The upper boundary for the setting values.
      getter: The method for getting the current value of the setting.
      setter: The method for setting the current value of the setting.
      default: The default value to assign to the setting.
      step: The step value for the variation of the setting values.
      
    .. versionadded:: 2.0.0 add *step* argument
    """

    # Ensuring that the two bounds are not equal
    if lowest == highest:
      raise ValueError("The two given bounds are equal !")

    # Ensuring that the given bounds are in the correct order
    if lowest > highest:
      self.log(logging.WARNING, f"Lowest ({lowest}) higher than highest "
                                f"({highest}), swapping them !")
      lowest, highest = highest, lowest

    self.lowest = lowest
    self.highest = highest
    self.step = step
    self.type = int if isinstance(self.lowest + self.highest, int) else float

    # Ensuring that the default value lies between the bounds
    if default is not None:
      if not lowest <= default <= highest:
        self.log(logging.WARNING,
                 f"The given default {default} is not between the lowest "
                 f"({lowest}) and highest ({highest}) values ! Setting to the "
                 f"center of the interval instead")
        default = self.type((self.lowest + self.highest) / 2)
      else:
        default = self.type(default)
    else:
      default = self.type((self.lowest + self.highest) / 2)

    super().__init__(name, getter, setter, default)

    self._check_default()

  @property
  def value(self) -> NbrType:
    """Returns the current value of the setting, by calling the getter if one
    was provided or else by returning the stored value.

    When the getter is called, calls the setter if one was provided and updates
    the sored value. After calling the setter, checks that the value was set
    by calling the getter and displays a warning message if the target and
    actual values don't match.
    """

    if self._getter is not None:
      return self.type(min(max(self._getter(), self.lowest), self.highest))
    else:
      return self.type(self._value_no_getter)

  @value.setter
  def value(self, val: NbrType) -> None:
    val = min(max(val, self.lowest), self.highest)
    self.log(logging.DEBUG, f"Setting the setting {self.name} to {val}")
    self.was_set = True

    self._value_no_getter = self.type(val)
    if self._setter is not None:
      self._setter(self.type(val))

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

  def reload(self,
             lowest: NbrType,
             highest: NbrType,
             value: Optional[NbrType] = None,
             default: Optional[NbrType] = None,
             step: Optional[NbrType] = None) -> None:
    """Allows modifying the limits and the step of the scale bar once it is
    already instantiated.

    Args:
      lowest: The new lowest possible value for the scale setting.
      highest: The new highest possible value for the scale setting.
      value: Optionally, a value to set the setting to when reloading it. If
        not provided, the current value remains if it is within the new bounds,
        otherwise the default is set.

        .. versionchanged:: 2.0.7 converted from mandatory to optional
      default: Optionally, the new default value for the setting. If not
        provided, the previous default remains, if it is still within the new
        bounds. Otherwise, a new default is defined.
      step: Optionally, a new step value for the setting.

    .. versionadded:: 2.0.0
    """

    self.log(logging.DEBUG, f"Reloading the setting {self.name}")

    # Ensuring that the two bounds are not equal
    if lowest == highest:
      raise ValueError("The two given bounds are equal !")

    # Ensuring that the given bounds are in the correct order
    if lowest > highest:
      self.log(logging.WARNING, f"Lowest ({lowest}) higher than highest "
                                f"({highest}), swapping them !")
      lowest, highest = highest, lowest

    # Updating the lowest, highest, step and default values
    self.lowest = lowest
    self.highest = highest
    self.step = step

    # Ensuring that the default value lies between the new bounds
    if default is None and not lowest <= self.default <= highest:
      self.log(logging.WARNING,
               f"The current default {self.default} is not between the lowest "
               f"({lowest}) and highest ({highest}) values ! Setting to the "
               f"center of the interval instead")
      self.default = self.type((self.lowest + self.highest) / 2)
    if default is not None:
      if not lowest <= default <= highest:
        self.log(logging.WARNING,
                 f"The given default {default} is not between the lowest "
                 f"({lowest}) and highest ({highest}) values ! Setting to the "
                 f"center of the interval instead")
        self.default = self.type((self.lowest + self.highest) / 2)
      else:
        self.default = default

    self._check_default()

    if value is not None and not self.lowest <= value <= self.highest:
      self.log(logging.WARNING,
               f"The given value {value} is not between the lowest "
               f"({lowest}) and highest ({highest}) values ! Ignoring it")
      value = None

    if value is None and not self.lowest <= self.value <= self.highest:
      self.log(logging.WARNING,
               f"The current value {self.value} is no longer between the "
               f"lowest ({lowest}) and highest ({highest}) values ! Setting "
               f"it to {self.default} instead")
      value = self.default

    # Updating the slider limits and the setting value
    if self.tk_obj is not None:
      self.tk_obj.configure(to=self.highest,
                            from_=self.lowest,
                            resolution=self.step)

    if value is not None:
      if self.tk_var is None:
        # If the setting was never set, not setting it yet but tweaking its
        # default so that it will only be set to the right value when expected
        if not self.was_set:
          self.log(logging.DEBUG, f"Setting default to {value} as a hack to "
                                  f"maintain settings call order")
          self.default = value
        # If the setting was already set though a kwarg, the user probably
        # doesn't want it silently overridden by a reload() call
        elif self.user_set and value != self.value:
          raise ValueError(f"Setting {self.name} forcibly set to value "
                           f"{self.value} using a kwarg, cannot override to "
                           f"{value} !")
        # Setting to the indicated value if parameter was previously set to
        # default
        else:
          self.value = value
      # Once in the graphical interface it is assumed that the user does not
      # want strict control over settings, always setting
      else:
        self.value = value

  def _check_default(self) -> None:
    """Checks if the step value is compatible with the limit values and
    types of the scale settings.
    
    .. versionadded:: 2.0.0
    """

    if self.step is not None:
      if self.type == int and isinstance(self.step, float):
        self.step = max(int(self.step), 1)
        self.log(logging.WARNING, f"Could not set {self.name} step "
                                  f"(lowest: int, step: float), "
                                  f"the step is now {self.step} !")
      if self.highest > self.lowest and self.step > self.highest - self.lowest:
        self.step = 1 if self.type == int else (self.highest -
                                                self.lowest) / 1000
        self.log(logging.WARNING, f"Could not set {self.name} step, "
                                  f"the step is now {self.step} !")
      if self.type == int and (self.highest - self.lowest) % self.step:
          self.highest -= (self.highest - self.lowest) % self.step
          self.log(logging.WARNING, f"Could not set {self.name} highest "
                                    f"with this step {self.step},"
                                    f" the highest is now {self.highest} !")
    else:
      self.step = 1 if self.type == int else (self.highest -
                                              self.lowest) / 1000
