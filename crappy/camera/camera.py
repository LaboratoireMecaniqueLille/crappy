# coding: utf-8

from typing import Callable, Optional, Tuple, Union, Any, Dict
import numpy as np

from .._global import DefinitionError

nbr_type = Union[int, float]


class MetaCam(type):
  """Metaclass ensuring that two cameras don't have the same name, and that all
  cameras define the required methods."""

  classes = {}

  needed_methods = ["get_image", "open", "close"]

  def __new__(mcs, name: str, bases: tuple, dct: dict) -> type:
    return super().__new__(mcs, name, bases, dct)

  def __init__(cls, name: str, bases: tuple, dct: dict) -> None:
    super().__init__(name, bases, dct)

    # Checking that a Camera with the same name doesn't already exist
    if name in cls.classes:
      raise DefinitionError(f"The {name} class is already defined !")

    # Gathering all the defined methods
    defined_methods = list(dct.keys())
    defined_methods += [base.__dict__.keys() for base in bases]

    # Checking for missing methods
    missing_methods = [meth for meth in cls.needed_methods
                       if meth not in defined_methods]

    # Raising if there are unexpected missing methods
    if missing_methods and name != "Camera":
      raise DefinitionError(f'Class {name} is missing the required method(s): '
                            f'{", ".join(missing_methods)}')

    # Otherwise, saving the class
    cls.classes[name] = cls


class Cam_setting:
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
    self._value_no_getter = val
    if self._setter is not None:
      self._setter(val)

    if self.value != val:
      print(f'[Cam settings] Could not set {self.name} to {val}, the value '
            f'is {self.value} !')


class Cam_bool_setting(Cam_setting):
  """Camera setting that can only be :obj:`True` or :obj:`False`."""

  def __init__(self,
               name: str,
               getter: Optional[Callable[[], bool]] = None,
               setter: Optional[Callable[[bool], None]] = None,
               default: bool = True) -> None:
    """Sets the attributes.

    Args:
      name: The name of the setting, that will be displayed in the GUI.
      getter: The method for getting the current value of the setting.
      setter: The method for setting the current value of the setting.
      default: The default value to assign to the setting.
    """

    super().__init__(name, getter, setter, default)


class Cam_scale_setting(Cam_setting):
  """Camera setting that can take any value between a lower and an upper
  boundary.

  This class can handle settings that should only take :obj:`int` values as
  well as settings that can take :obj:`float` value.
  """

  def __init__(self,
               name: str,
               lowest: nbr_type,
               highest: nbr_type,
               getter: Optional[Callable[[], nbr_type]] = None,
               setter: Optional[Callable[[nbr_type], None]] = None,
               default: Optional[nbr_type] = None) -> None:
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
  def value(self) -> nbr_type:
    """Returns the current value of the setting, by calling the getter if one
    was provided or else by returning the stored value."""

    if self._getter is not None:
      return self.type(min(max(self._getter(), self.lowest), self.highest))
    else:
      return self.type(self._value_no_getter)

  @value.setter
  def value(self, val: nbr_type) -> None:
    val = min(max(val, self.lowest), self.highest)

    self._value_no_getter = self.type(val)
    if self._setter is not None:
      self._setter(self.type(val))

    if self.value != val:
      print(f'[Cam settings] Could not set {self.name} to {val}, the value '
            f'is {self.value} !')


class Cam_choice_setting(Cam_setting):
  """Camera setting that can take any value from a predefined list of
  values."""

  def __init__(self,
               name: str,
               choices: Tuple[str, ...],
               getter: Optional[Callable[[], str]] = None,
               setter: Optional[Callable[[str], None]] = None,
               default: Optional[str] = None) -> None:
    """Sets the attributes.

    Args:
      name: The name of the setting, that will be displayed in the GUI.
      choices: A tuple listing the possible values for the setting.
      getter: The method for getting the current value of the setting.
      setter: The method for setting the current value of the setting.
      default: The default value to assign to the setting.
    """

    self.choices = choices

    if default is None:
      default = choices[0]

    super().__init__(name, getter, setter, default)


class Camera(metaclass=MetaCam):
  """Base class for every camera object.

  It contains all the methods shared by these classes and sets MetaCam as their
  metaclass.
  """

  def __init__(self) -> None:
    """Simply sets the dict containing the settings."""

    self.settings: Dict[str, Cam_setting] = dict()

  def read_image(self) -> Tuple[float, np.ndarray]:
    """To be removed, temporarily ensures the compatibility with the blocks
    that haven't been updated yet."""

    return self.get_image()

  def add_bool_setting(self,
                       name: str,
                       getter: Optional[Callable[[], bool]] = None,
                       setter: Optional[Callable[[bool], None]] = None,
                       default: bool = True) -> None:
    """Adds a boolean setting, whose value is either :obj:`True` or
    :obj:`False`.

    Args:
      name: The name of the setting, that will be displayed in the GUI and can
        be used to directly get the value of the setting by calling
        ``self.<name>``
      getter: The method for getting the current value of the setting. If not
        given, the returned value is simply the last one that was set.
      setter: The method for setting the current value of the setting. If not
        given, the value to be set is simply stored.
      default: The default value to assign to the setting.
    """

    if name in self.settings:
      raise ValueError('This setting already exists !')
    self.settings[name] = Cam_bool_setting(name, getter, setter, default)

  def add_scale_setting(self,
                        name: str,
                        lowest: nbr_type,
                        highest: nbr_type,
                        getter: Optional[Callable[[], nbr_type]] = None,
                        setter: Optional[Callable[[nbr_type], None]] = None,
                        default: Optional[nbr_type] = None) -> None:
    """Adds a scale setting, whose value is an :obj:`int` or a :obj:`float`
    clamped between two boundaries.

    Note:
      If any of ``lowest`` or ``highest`` is a :obj:`float`, then the setting
      is considered to be of type float and can take float values. Otherwise,
      it is considered of type int and can only take integer values.

    Args:
      name: The name of the setting, that will be displayed in the GUI and can
        be used to directly get the value of the setting by calling
        ``self.<name>``
      lowest: The lowest possible value for the setting.
      highest: The highest possible value for the setting.
      getter: The method for getting the current value of the setting. If not
        given, the returned value is simply the last one that was set.
      setter: The method for setting the current value of the setting. If not
        given, the value to be set is simply stored.
      default: The default value to assign to the setting. If not given, will
        be the average of ``lowest`` and ``highest``.
    """

    if name in self.settings:
      raise ValueError('This setting already exists !')
    self.settings[name] = Cam_scale_setting(name, lowest, highest, getter,
                                            setter, default)

  def add_choice_setting(self,
                         name: str,
                         choices: Tuple[str, ...],
                         getter: Optional[Callable[[], str]] = None,
                         setter: Optional[Callable[[str], None]] = None,
                         default: Optional[str] = None) -> None:
    """Adds a choice setting, that can take a limited number of predefined
    :obj:`str` values.

    Args:
      name: The name of the setting, that will be displayed in the GUI and can
        be used to directly get the value of the setting by calling
        ``self.<name>``
      choices: A :obj:`tuple` containing the possible values for the setting.
      getter: The method for getting the current value of the setting. If not
        given, the returned value is simply the last one that was set.
      setter: The method for setting the current value of the setting. If not
        given, the value to be set is simply stored.
      default: The default value to assign to the setting. If not given, will
        be the fist item in ``choices``.
    """

    if name in self.settings:
      raise ValueError('This setting already exists !')
    self.settings[name] = Cam_choice_setting(name, choices, getter, setter,
                                             default)

  def set_all(self, **kwargs) -> None:
    """Checks if the kwargs are valid, sets them, and for settings that are not
    in kwargs sets them to their default value."""

    unexpected = tuple(kwarg for kwarg in kwargs if kwarg not in self.settings)
    if unexpected:
      raise ValueError(f'Unexpected argument(s) {", ".join(unexpected)} for '
                       f'camera {type(self).__name__}.')

    for name, setting in self.settings.items():
      setting.value = kwargs[name] if name in kwargs else setting.default

  def __getattr__(self, item: str) -> Any:
    """Method for getting the value of a setting directly by calling
    ``self.<setting name>``.

    It is called in case __getattribute__ doesn't work properly, and tries to
    return the corresponding setting value."""

    try:
      return self.settings[item].value
    except (AttributeError, KeyError):
      raise AttributeError(f'No attribute nor setting named {item}')

  def __setattr__(self, key: str, val: Any) -> None:
    """Method for setting the value of a setting directly by calling
    ``self.<setting name> = <value>``."""

    if key != 'settings' and key in self.settings:
      self.settings[key].value = val
    else:
      super().__setattr__(key, val)
