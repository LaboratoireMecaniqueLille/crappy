# coding: utf-8

from typing import Callable, Optional, Tuple, Union, Any, Dict
from time import sleep
import numpy as np
from multiprocessing import current_process
import logging

from .meta_camera import MetaCamera
from .camera_setting import CameraSetting, CameraBoolSetting, \
  CameraScaleSetting, CameraChoiceSetting

NbrType = Union[int, float]


class Camera(metaclass=MetaCamera):
  """Base class for every camera object.

  It contains all the methods shared by these classes and sets MetaCam as their
  metaclass.
  """

  def __init__(self, *_, **__) -> None:
    """Simply sets the dict containing the settings and the name of the
    trigger setting."""

    self.settings: Dict[str, CameraSetting] = dict()
    self.trigger_name = 'Trigger'
    self._logger: Optional[logging.Logger] = None

  def log(self, level: int, msg: str) -> None:
    """"""

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)

  def open(self, **kwargs) -> None:
    """This method should initialize the connection to the camera, configure
    the camera, and start the image acquisition.

    This method also takes as arguments all the kwargs that were passed to the
    Camera block but not used by it. Some may be used directly, e.g. for
    choosing which camera to open out of several possible ones, and the others
    should indicate values to set for available settings. It is fine not to
    provide any values for the settings here, as each setting has a default.

    To effectively set the setting values, the method :meth:`set_all` has to
    be called at the end of open (e.g. ``self.set_all(**kwargs)``). This is
    true even if no value to set was given in the kwargs. If it is not called,
    the settings won't actually be set on the camera.

    If some camera settings require values from the camera for their
    instantiation (e.g.
    ``self.add_setting(..., highest=self.cam.max_width(), ...)``), they should
    be instantiated here. And of course before calling :meth:`set_all`.
    """

    self.set_all(**kwargs)

  def get_image(self) -> Optional[Union[Tuple[Dict[str, Any], float],
                                        np.ndarray]]:
    """Acquires an image and returns it along with its metadata or timestamp.

    It is also fine for this method to return :obj:`None`. The image should be
    returned as a numpy array, and the metadata as a :obj:`dict` or the
    timestamp as a :obj:`float`. The keys of the metadata dictionary should
    preferably be valid Exif tags, so that the metadata can be embedded into
    the image file when saving.

    In order for the recording of images to run, the metadata dict must
    contain at least the ``'t(s)'`` and ``''ImageUniqueID''`` keys, whose
    values should be the timestamp when the frame was acquired (as returned by
    ``time.time()``) and the frame number as an :obj:`int`.
    """

    self.log(logging.WARNING, "The get_img method was called but is not "
                              "defined !\nNo image can be acquired if this "
                              "method isn't defined")
    sleep(1)
    return

  def close(self) -> None:
    """This method should perform any action required for properly stopping the
    image acquisition and/or closing the connection to the camera.

    This step is usually extremely important in order for the camera resources
    to be released. Otherwise, it might be impossible to re-open the camera
    from Crappy without resetting the hardware connection with it.
    """

    ...

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

    # Checking if the given name is valid
    if name == self.trigger_name:
      raise ValueError(f"The name {self.trigger_name} is reserved for the "
                       f"trigger setting !")
    if name in self.settings:
      raise ValueError('This setting already exists !')
    self.log(logging.INFO, f"Adding the {name} bool setting")
    self.settings[name] = CameraBoolSetting(name, getter, setter, default)

  def add_scale_setting(self,
                        name: str,
                        lowest: NbrType,
                        highest: NbrType,
                        getter: Optional[Callable[[], NbrType]] = None,
                        setter: Optional[Callable[[NbrType], None]] = None,
                        default: Optional[NbrType] = None) -> None:
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

    # Checking if the given name is valid
    if name == self.trigger_name:
      raise ValueError(f"The name {self.trigger_name} is reserved for the "
                       f"trigger setting !")
    if name in self.settings:
      raise ValueError('This setting already exists !')
    self.log(logging.INFO, f"Adding the {name} scale setting")
    self.settings[name] = CameraScaleSetting(name, lowest, highest, getter,
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

    # Checking if the given name is valid
    if name == self.trigger_name:
      raise ValueError(f"The name {self.trigger_name} is reserved for the "
                       f"trigger setting !")
    if name in self.settings:
      raise ValueError('This setting already exists !')
    self.log(logging.INFO, f"Adding the {name} choice setting")
    self.settings[name] = CameraChoiceSetting(name, choices, getter, setter,
                                              default)

  def add_trigger_setting(self,
                          getter: Optional[Callable[[], str]] = None,
                          setter: Optional[Callable[[str], None]] = None
                          ) -> None:
    """Adds a specific choice setting for controlling the trigger mode of the
    camera. The reserved name for this setting is ``'Trigger'``.

    This setting is mainly intended for cameras that can run either in free run
    mode or in hardware trig mode. The three possible choices for this setting
    are : ``'Free run'``, ``'Hdw after config'`` and ``'Hardware'``. Default is
    ``'Free run'``.

    The setter method is expected to set the camera to free run mode in
    ``'Free run'`` and ``'Hdw after config'`` choices, and to hardware trigger
    mode in the ``'Hardware'`` choice. The getter method should return either
    ``'Free run'`` or ``'Hdw after config'`` in free run mode, depending on the
    last set value for the setting, and ``'Hardware'`` in hardware trig mode.
    It can also be left to :obj:`None`.

    The rationale behind the ``'Hdw after config'`` choice is to allow the user
    to tune settings in the configuration window with the camera in free run
    mode, and to switch afterwards to the hardware trigger mode for the actual
    test. It proves extremely useful if the hardware triggers are generated
    from Crappy, as they're not started yet when the configuration window is
    running.

    Args:
      getter: The method for getting the current value of the setting. If not
        given, the returned value is simply the last one that was set.
      setter: The method for setting the current value of the setting. If not
        given, the value to be set is simply stored.
    """

    if self.trigger_name in self.settings:
      raise ValueError("There can only be one trigger setting per camera !")

    self.log(logging.INFO, f"Adding the {self.trigger_name} trigger setting")
    self.settings[self.trigger_name] = CameraChoiceSetting(
      name=self.trigger_name, choices=('Free run',
                                       'Hdw after config',
                                       'Hardware'),
      getter=getter, setter=setter, default='Free run')

  def set_all(self, **kwargs) -> None:
    """Checks if the kwargs are valid, sets them, and for settings that are not
    in kwargs sets them to their default value."""

    unexpected = tuple(kwarg for kwarg in kwargs if kwarg not in self.settings)
    if unexpected:
      raise ValueError(f'Unexpected argument(s) {", ".join(unexpected)} for '
                       f'camera {type(self).__name__}.')

    self.log(logging.INFO, "Setting all the setting values")
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
