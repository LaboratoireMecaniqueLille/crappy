# coding: utf-8

# Todo
#   The interface doesn't update when setting the parameters as arguments

from time import time, sleep
from typing import Callable, Union, Any

from .._global import DefinitionError


class MetaCam(type):
  """Metaclass that will define all cameras.

  Note:
    Camera classes should be of this type.

    To do so, simply add ``__metaclass__ = MetaCam`` in the class definition.
    (Obviously, you must import this Metaclass first.)

    :class:`MetaCam` is a MetaClass: We will NEVER do ``c = MetaCam(...)``.

    The :meth:`__init__` is used to init the classes of type MetaCam (with
    ``__metaclass__ = MetaCam`` as a class attribute) and NOT an instance of
    MetaClass.
  """

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
  """This class represents an attribute of the camera that can be set."""

  def __init__(self,
               name: str,
               getter: Callable,
               setter: Callable,
               limits: Union[None, tuple, bool, dict],
               default: Any) -> None:
    """Sets the instance attributes.

    Args:
      name: The name of the setting.
      default: The default value, if not specified it will be set to this
        value.
      getter: Function to read this value from the device. If set to
        :obj:`None`, it will assume that the setting always happened correctly.
      setter: A function that will be called when setting the parameter to a
        new value. Can do nothing, it will only change its value and nothing
        else.
      limits: It contains the available values for this parameter.

        The possible limit types are:

          - :obj:`None`: Values will not be tested and the parameter will not
            appear in CameraConfig.

          - A :obj:`tuple` of 2 :obj:`int` or :obj:`float`: Values must be
            between first and second value. CameraConfig will add a scale
            widget to set it. If they are integers, all the integers between
            them will be accessible, if they are floats, the range will be
            divided in 1000 in the scale widget.

            Note that if the upper value is callable (a function or method), it
            will be set to the return value of this function. It allows reading
            the max value from the device.

          - A :obj:`bool`: Possible values will be :obj:`True` or :obj:`False`,
            CameraConfig will add a checkbox to edit the value (default can be
            :obj:`True` or :obj:`False`, it doesn't matter).

          - A :obj:`dict`: Possible values are the values of the dict,
            CameraConfig will add radio buttons showing the keys, to set it to
            the corresponding value.
    """

    self.name = name
    self.getter = getter
    self.setter = setter
    self.limits = limits
    self.default = default
    self._value = None
    if getter is None:
      self._value = default
      self.getter = lambda *args: self._value

  @property
  def value(self) -> Any:
    if self._value is None:
      self._value = self.getter()
      if type(self.limits) is tuple and callable(self.limits[1]):
          self.limits = (self.limits[0], self.limits[1]())
      if self.default is None:
        self.default = self._value
    return self._value

  # Here is the interesting part: When we set value (setting.value = x),
  # we will go through all of this, and the new value will be the actual
  # value of the setting after the operation
  @value.setter
  def value(self, i: Any) -> None:
    _ = self.value  # Detail: to make sure we called value getter once
    # if type(self.limits) == tuple:
    #  if not self.limits[0] <= i <= self.limits[1]:
    #    print("[Cam_setting] Parameter", i, "out of range ", self.limits)
    #    return
    if isinstance(self.limits, dict):
      if i not in self.limits.values():
        print("[Cam_setting] Parameter", i, "not available", self.limits)
        return
    elif isinstance(self.limits, bool):
      i = bool(i)
    # We could actually wait to see if setter is successful before setting the
    # value, but if setter uses self.parameter, it will still be set to its old
    # value until it returns...
    self.setter(i)
    self._value = i
    new_val = self.getter()
    if new_val != i:
      print("[Cam_setting] Could not set", self.name, "to", i,
            "value is", new_val)
    self._value = new_val

  def __str__(self) -> str:
    if self.limits:
      return "Setting: " + str(self.name) + ", value:" + str(self._value) + \
             " Limits:" + str(self.limits)
    else:
      return "Setting: " + str(self.name) + ", value:" + str(self._value)

  def __repr__(self) -> str:
    return self.__str__()


class Camera(metaclass=MetaCam):
  """This class represents a camera sensor.

  It may have settings: They represent all that can be set on the camera:
  height, width, exposure, AEAG, external trigger, etc...

  Note:
    Each parameter is represented by a :class:`Cam_setting` object: it includes
    the default value, a function to set and get parameter, etc...

    This class makes it transparent to the user: you can access a setting by
    using ``myinstance.setting = stuff``.

    It will automatically check the validity and try to set it (see
    :class:`Cam_setting`).

    Don't forget to call the :meth:`__init__` in the children or
    :meth:`__getattr__` will fall in an infinite recursion loop looking for
    settings...
  """

  def __init__(self) -> None:
    self.settings = {}
    self.last = time()
    self.max_fps = None
    self.name = "Camera"

  @property
  def max_fps(self) -> float:
    return self._max_fps

  @max_fps.setter
  def max_fps(self, value: float) -> None:
    """To compute :attr:`self.delay` again when fps is set."""

    self._max_fps = value
    if value:
      self.delay = 1/value
    else:
      self.delay = 0

  def add_setting(self,
                  name: str,
                  getter: Callable = None,
                  setter: Callable = lambda *val: None,
                  limits: Union[None, tuple, bool, dict] = None,
                  default: Any = None) -> None:
    """Wrapper to simply add a new setting to the camera."""

    assert name not in self.settings, "This setting already exists"
    self.settings[name] = Cam_setting(name, getter, setter, limits, default)

  @property
  def available_settings(self) -> list:
    """Returns a :obj:`list` of available settings."""

    return [x.name for x in list(self.settings.values())] + ["max_fps"]

  @property
  def settings_dict(self) -> dict:
    """Returns settings as a :obj:`dict`, keys are the names of the settings
    and values are `setting.value`."""

    d = dict(self.settings)
    for k in d:
      d[k] = d[k]._value
    return d

  def set_all(self, override: bool = False, **kwargs) -> None:
    """Sets all the settings based on `kwargs`.

    Note:
      If not specified, the setting will take its default value.

      If override is :obj:`True`, it will not assume a setting and reset it
      unless it is already default.
    """

    for s in self.settings:
      if s in kwargs:
        if self.settings[s].value != kwargs[s] or override:
          self.settings[s].value = kwargs[s]
        else:
          pass
        del kwargs[s]
      elif self.settings[s].value != self.settings[s].default:
        self.settings[s].value = self.settings[s].default
      else:
        pass
    for k, v in kwargs.items():
      setattr(self, k, v)

  def reset_all(self) -> None:
    """Reset all the settings to their default values."""

    self.set_all()

  def read_image(self) -> tuple:
    """This method is a wrapper for :meth:`get_image` that will limit fps to
    `max_fps`."""

    if self.delay:
      t = time()
      wait = self.last - t + self.delay
      while wait > 0:
        t = time()
        wait = self.last - t + self.delay
        sleep(max(0., wait / 10))
      self.last = t
    return self.get_image()

  def __getattr__(self, i: str) -> Any:
    """The idea is simple: if the camera has this attribute: return it (default
    behavior) else, try to find the corresponding setting and return its value.

    Note that we made sure to raise an :exc:`AttributeError` if it is neither a
    camera attribute nor a setting.

    Example:
      If Camera definition contains ``self.add_setting("width",1280,set_w)``,
      and ``cam = Camera()``, then ``cam.width`` will return `1280`.
    """

    try:
      return self.__getattribute__(i)
    except AttributeError:
      try:
        return self.settings[i].value
      except KeyError:
        raise AttributeError("No such attribute: " + i)
      except RuntimeError:
        print("You have probably forgotten to call Camera.__init__(self)!")
        raise AttributeError("No such attribute:" + i)

  def __setattr__(self, attr: str, val: Any) -> None:
    """Same as :meth:`__getattr__`: if it is a setting, then set its value
    using the setter in the :class:`Cam_setting`, else use the default
    behavior.

    It is important to make sure we don't try to set 'settings', it would
    recursively call :meth:`__getattr__` and enter an infinite loop, hence the
    condition.

    Example:
      ``cam.width = 2048`` will be like ``cam.settings['width'].value = 2048``.
      It allows for simple settings of the camera.
    """

    if attr != "settings" and attr in self.settings:
      self.settings[attr].value = val
    else:
      super(Camera, self).__setattr__(attr, val)

  def __str__(self) -> str:
    return self.name + " camera with {} settings".format(len(self.settings))

  def __repr__(self) -> str:
    s = self.__str__()
    for i in self.settings.values():
      s += ("\n" + str(i))
    return s
