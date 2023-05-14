# coding: utf-8

from typing import Callable, Optional, Tuple, Union, Any, Dict, Iterable
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

    # These names are reserved for special settings
    self.trigger_name = 'trigger'
    self.roi_x_name = 'ROI_x'
    self.roi_y_name = 'ROI_y'
    self.roi_width_name = 'ROI_width'
    self.roi_height_name = 'ROI_height'
    self._soft_roi_set = False
    self._reserved = (self.trigger_name, self.roi_x_name, self.roi_y_name,
                      self.roi_width_name, self.roi_height_name)

    self._logger: Optional[logging.Logger] = None

  def log(self, level: int, msg: str) -> None:
    """Records log messages for the Camera.

    Also instantiates the logger when logging the first message.

    Args:
      level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.
    """

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

    If a configuration window is used, it will be possible to set this setting
    by (un)checking a checkbox.

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
    if name in self._reserved:
      raise ValueError(f"The name {name} is reserved for a different type of "
                       f"setting ! !")
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

    If a configuration window is used, it will be possible to set this setting
    by moving a slider.

    Note:
      If any of ``lowest`` or ``highest`` is a :obj:`float`, then the setting
      is considered to be of type float and can take float values. Otherwise,
      it is considered of type :obj:`int` and can only take integer values.

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
    if name in self._reserved:
      raise ValueError(f"The name {name} is reserved for a different type of "
                       f"setting ! !")
    if name in self.settings:
      raise ValueError('This setting already exists !')
    self.log(logging.INFO, f"Adding the {name} scale setting")
    self.settings[name] = CameraScaleSetting(name, lowest, highest, getter,
                                             setter, default)

  def add_choice_setting(self,
                         name: str,
                         choices: Iterable[str],
                         getter: Optional[Callable[[], str]] = None,
                         setter: Optional[Callable[[str], None]] = None,
                         default: Optional[str] = None) -> None:
    """Adds a choice setting, that can take a limited number of predefined
    :obj:`str` values.

    If a configuration window is used, it will be possible to set this setting
    by selecting one of several possible radio buttons.

    Args:
      name: The name of the setting, that will be displayed in the GUI and can
        be used to directly get the value of the setting by calling
        ``self.<name>``
      choices: An iterable (like a :obj:`tuple` or a :obj:`list`) containing
        the possible values for the setting.
      getter: The method for getting the current value of the setting. If not
        given, the returned value is simply the last one that was set.
      setter: The method for setting the current value of the setting. If not
        given, the value to be set is simply stored.
      default: The default value to assign to the setting. If not given, will
        be the fist item in ``choices``.
    """

    # Checking if the given name is valid
    if name in self._reserved:
      raise ValueError(f"The name {name} is reserved for a different type of "
                       f"setting ! !")
    if name in self.settings:
      raise ValueError('This setting already exists !')
    self.log(logging.INFO, f"Adding the {name} choice setting")
    self.settings[name] = CameraChoiceSetting(name, tuple(choices), getter,
                                              setter, default)

  def add_trigger_setting(self,
                          getter: Optional[Callable[[], str]] = None,
                          setter: Optional[Callable[[str], None]] = None
                          ) -> None:
    """Adds a specific choice setting for controlling the trigger mode of the
    camera. The reserved name for this setting is ``'trigger'``.

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
    mode, and to switch afterward to the hardware trigger mode for the actual
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

    self.log(logging.INFO, f"Adding the '{self.trigger_name}' trigger setting")
    self.settings[self.trigger_name] = CameraChoiceSetting(
      name=self.trigger_name, choices=('Free run',
                                       'Hdw after config',
                                       'Hardware'),
      getter=getter, setter=setter, default='Free run')

  def add_software_roi(self, width: int, height: int) -> None:
    """Creates the settings needed for setting a software ROI.

    The ROI is a rectangular area defining which part of the image to keep. It
    can be tuned by the user by setting the position of the upper left corner,
    as well as the width and the height.

    Using a ROI reduces the size of the image for processing, displaying and
    saving, which improves the overall performance.

    Args:
      width: The width of the acquired images, in pixels.
      height: The height of the acquired images, in pixels.
    """

    # Checking that the software ROI setting does not already exist
    if self._soft_roi_set:
      raise ValueError("There can only be one set of software settings per "
                       "camera !")

    # Instantiating the CameraSetting objects
    self.log(logging.INFO, "Adding the software ROI settings")
    self.settings[self.roi_x_name] = CameraScaleSetting(
        name=self.roi_x_name, lowest=0, highest=width-1, getter=None,
        setter=None, default=0)
    self.settings[self.roi_y_name] = CameraScaleSetting(
        name=self.roi_y_name, lowest=0, highest=height - 1, getter=None,
        setter=None, default=0)
    self.settings[self.roi_width_name] = CameraScaleSetting(
        name=self.roi_width_name, lowest=1, highest=width, getter=None,
        setter=None, default=width)
    self.settings[self.roi_height_name] = CameraScaleSetting(
        name=self.roi_height_name, lowest=1, highest=height, getter=None,
        setter=None, default=height)

    self._soft_roi_set = True

  def reload_software_roi(self, width: int, height: int) -> None:
    """Updates the software ROI boundaries when the width and/or the height of
    the acquired images change.

    Args:
      width: The width of the acquired images, in pixels.
      height: The height of the acquired images, in pixels.
    """

    if self._soft_roi_set:
      self.log(logging.DEBUG, "Reloading the software ROI settings")
      self.settings[self.roi_x_name].reload(lowest=0, highest=width-1,
                                            default=0)
      self.settings[self.roi_y_name].reload(lowest=0, highest=height - 1,
                                            default=0)
      self.settings[self.roi_width_name].reload(lowest=1, highest=width,
                                                default=width)
      self.settings[self.roi_height_name].reload(lowest=1, highest=height,
                                                 default=height)
      self.settings[self.roi_x_name].value = 0
      self.settings[self.roi_y_name].value = 0
      self.settings[self.roi_width_name].value = width
      self.settings[self.roi_height_name].value = height
    else:
      self.log(logging.WARNING, "Cannot reload the software ROI settings as "
                                "they are not defined !")

  def apply_soft_roi(self, img: np.ndarray) -> Optional[np.ndarray]:
    """Takes an image as an input, and crops according to the selected software
    ROI dimensions.

    Might return :obj:`None` in case there's no pixel left on the cropped
    image. Returns the original image if the software ROI settings are not
    defined.
    """

    if self._soft_roi_set:
      x = self.settings[self.roi_x_name].value
      y = self.settings[self.roi_y_name].value
      width = self.settings[self.roi_width_name].value
      height = self.settings[self.roi_height_name].value

      # Cropping to the requested size
      img = img[y:y+height, x:x+width]
      if img.size:
        return img
      # If there's no pixel left to display, return None
      else:
        return

    # Simply returning the image if the ROI settings were not defined
    else:
      return img

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

    It is called in case :meth:`__getattribute__` doesn't work properly, and
    tries to return the corresponding setting value."""

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
