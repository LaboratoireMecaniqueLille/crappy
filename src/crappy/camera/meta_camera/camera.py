# coding: utf-8

from typing import Optional, Union, Any
from collections.abc import Callable, Iterable
from time import sleep
import numpy as np
from multiprocessing import current_process
import logging

from .meta_camera import MetaCamera
from .camera_setting import CameraSetting, CameraBoolSetting, \
  CameraScaleSetting, CameraChoiceSetting

NbrType = Union[int, float]


class Camera(metaclass=MetaCamera):
  """Base class for every Camera object. Implements methods shared by all the
  Cameras, and ensures their dataclass is :class:`~crappy.camera.MetaCamera`.

  The Camera objects are helper classes used by the
  :class:`~crappy.blocks.Camera` Block to interface with cameras.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self, *_, **__) -> None:
    """Simply sets the :obj:`dict` containing the settings, and the name of the
    reserved settings.

    Here, :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting` can
    be added to the camera. It can be done using one of the
    :meth:`add_bool_setting`, :meth:`add_scale_setting`,
    :meth:`add_choice_setting`, :meth:`add_trigger_setting`, or
    :meth:`add_software_roi` methods. Refer to the documentation of these
    methods for more information.
    
    .. versionchanged:: 2.0.0 now accepts *args* and *kwargs*
    """

    self.settings: dict[str, CameraSetting] = dict()

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
      
    .. versionadded:: 2.0.0
    """

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)

  def open(self, **kwargs) -> None:
    """This method should initialize the connection to the camera, configure
    the camera, and start the image acquisition.

    Here, :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting` can
    be added to the camera. It can be done using one of the
    :meth:`add_bool_setting`, :meth:`add_scale_setting`,
    :meth:`add_choice_setting`, :meth:`add_trigger_setting`, or
    :meth:`add_software_roi` methods. Refer to the documentation of these
    methods for more information. It is preferable to add the settings during
    :meth:`__init__`, but this is not always possible (e.g. if values read
    from the camera are required to instantiate a setting).

    This method takes as arguments all the kwargs that were passed to the
    :class:`~crappy.blocks.Camera` Block but not used by it. These kwargs can
    have two possible usages. They can be used to parametrize the method, e.g.
    a serial number can be given to choose which camera to open. Alternatively,
    they can be used for adjusting camera settings values. It is also fine not
    to provide any argument to this method.

    When providing a value for a setting, it should be done by giving the kwarg
    ``<setting_name>=<setting_value>`` to the Camera Block, with
    ``<setting_name>`` the exact name given to the setting. It can be desirable
    to provide setting values here in case the display of the
    :class:`~crappy.tool.camera_config.CameraConfig` is disabled
    (``config=False`` set on the Camera Block), or to gain time if the correct
    values are already known.

    Important:
      To effectively set the setting values, the method :meth:`set_all` must be
      called e.g. ``self.set_all(**kwargs)``). It is usually called at the very
      end of the method. This is true even if no value to set was given in the
      kwargs. If :meth:`set_all` is not called, the settings won't actually be
      set on the camera.
    
    .. versionadded:: 1.5.10
    """

    self.set_all(**kwargs)

  def get_image(self) -> Optional[tuple[Union[dict[str, Any], float],
                                        np.ndarray]]:
    """Acquires an image and returns it along with its metadata or timestamp.

    This method should return two objects, the second being the image as a
    :mod:`numpy` array. If the first object is a :obj:`float`, it is considered
    as the timestamp associated with the image (as returned by
    :obj:`time.time`). If it is a :obj:`dict`, it should contain the metadata
    associated with the image. It is also fine for this method to return
    :obj:`None` if no image could be acquired.

    If metadata is returned, it should contain at least the ``'t(s)'`` and
    ``'ImageUniqueID'`` keys, containing the timestamp as a :obj:`float` (as
    returned by :obj:`time.time`) and the frame number as an :obj:`int`. Any
    other field can be provided. This metadata is used by the
    :class:`~crappy.blocks.camera_processes.ImageSaver` class and saved along
    with the frames if the ``record_images`` argument of the
    :class:`~crappy.blocks.Camera` Block is :obj:`True`. The keys should
    preferably be valid EXIF tags, so that the information can be embedded in
    the recorded images.

    If only a timestamp is provided, a metadata :obj:`dict` is built internally
    but the user has no control over it.

    .. versionadded:: 1.5.10
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

    .. versionadded:: 1.5.10
    """

    ...

  def add_bool_setting(self,
                       name: str,
                       getter: Optional[Callable[[], bool]] = None,
                       setter: Optional[Callable[[bool], None]] = None,
                       default: bool = True) -> None:
    """Adds a boolean setting, whose value is either :obj:`True` or
    :obj:`False`.

    It creates an instance of
    :class:`~crappy.camera.meta_camera.camera_setting.CameraBoolSetting` using
    the provided arguments.

    If a :class:`~crappy.tool.camera_config.CameraConfig` window is displayed
    (``config=True`` set on the Camera Block), this setting will appear as a
    checkbox.

    Args:
      name: The name of the setting, that will be displayed in the
        configuration window and allows to access the setting directly with
        ``self.<name>``. Also the name to use for setting the value as a kwarg
        of the Camera Block (``<name>=<value>``).
      getter: The method for getting the current value of the setting. If not
        given, the returned value is simply the last one that was set.
      setter: The method for setting the current value of the setting. If not
        given, the value to be set is simply stored.
      default: The default value to assign to the setting.
    
    .. versionadded:: 1.5.10
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
                        default: Optional[NbrType] = None,
                        step: Optional[NbrType] = None) -> None:
    """Adds a scale setting, whose value is an :obj:`int` or a :obj:`float`
    lying between two boundaries.

    It creates an instance of
    :class:`~crappy.camera.meta_camera.camera_setting.CameraScaleSetting` using
    the provided arguments.

    If a :class:`~crappy.tool.camera_config.CameraConfig` window is displayed
    (``config=True`` set on the Camera Block), this setting will appear as a
    slider.

    Note:
      If any of ``lowest`` or ``highest`` is a :obj:`float`, then the setting
      is considered to be of type float and can take float values. Otherwise,
      it is considered of type :obj:`int` and can only take integer values.

    Args:
      name: The name of the setting, that will be displayed in the
        configuration window and allows to access the setting directly with
        ``self.<name>``. Also the name to use for setting the value as a kwarg
        of the Camera Block (``<name>=<value>``).
      lowest: The lowest possible value for the setting.
      highest: The highest possible value for the setting.
      getter: The method for getting the current value of the setting. If not
        given, the returned value is simply the last one that was set.
      setter: The method for setting the current value of the setting. If not
        given, the value to be set is simply stored.
      default: The default value to assign to the setting. If not given, will
        be the average of ``lowest`` and ``highest``.
      step: The step value for the variation of the setting values.

        .. versionadded:: 2.0.0

    .. versionadded:: 1.5.10
    """

    # Checking if the given name is valid
    if name in self._reserved:
      raise ValueError(f"The name {name} is reserved for a different type of "
                       f"setting ! !")
    if name in self.settings:
      raise ValueError('This setting already exists !')
    self.log(logging.INFO, f"Adding the {name} scale setting")
    self.settings[name] = CameraScaleSetting(name, lowest, highest, getter,
                                             setter, default, step)

  def add_choice_setting(self,
                         name: str,
                         choices: Iterable[str],
                         getter: Optional[Callable[[], str]] = None,
                         setter: Optional[Callable[[str], None]] = None,
                         default: Optional[str] = None) -> None:
    """Adds a choice setting, that can take a limited number of predefined
    :obj:`str` values.

    It creates an instance of
    :class:`~crappy.camera.meta_camera.camera_setting.CameraChoiceSetting`
    using the provided arguments.

    If a :class:`~crappy.tool.camera_config.CameraConfig` window is displayed
    (``config=True`` set on the Camera Block), this setting will appear as a
    set of radio buttons.

    Args:
      name: The name of the setting, that will be displayed in the
        configuration window and allows to access the setting directly with
        ``self.<name>``. Also the name to use for setting the value as a kwarg
        of the Camera Block (``<name>=<value>``).
      choices: An iterable (like a :obj:`tuple` or a :obj:`list`) containing
        the possible values for the setting.
      getter: The method for getting the current value of the setting. If not
        given, the returned value is simply the last one that was set.
      setter: The method for setting the current value of the setting. If not
        given, the value to be set is simply stored.
      default: The default value to assign to the setting. If not given, will
        be the fist item in ``choices``.

    .. versionadded:: 1.5.10
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
    """Adds a specific setting for controlling the trigger mode of the camera.
    The reserved name for this setting is ``'trigger'``.

    It creates an instance of
    :class:`~crappy.camera.meta_camera.camera_setting.CameraChoiceSetting`
    using a reserved name, and predefined choices and default.

    This setting is mainly intended for cameras that can run either in free run
    mode or in hardware trig mode. The three possible choices for this setting
    are :
    ::

      'Free run', 'Hdw after config', 'Hardware'

    Default is ``'Free run'``.

    The setter method is expected to set the camera to free run mode in
    ``'Free run'`` and ``'Hdw after config'`` choices, and to hardware trigger
    mode in the ``'Hardware'`` choice. The getter method should return either
    ``'Free run'`` or ``'Hdw after config'`` in free run mode, depending on the
    last set value for the setting, and ``'Hardware'`` in hardware trig mode.
    It can also be left to :obj:`None`.

    The rationale behind the ``'Hdw after config'`` choice is to allow the user
    to tune settings in the :class:`~crappy.tool.camera_config.CameraConfig`
    window with the camera in free run mode, and to switch afterward to the
    hardware trigger mode once the test begins. It proves extremely useful if
    the hardware triggers are generated from Crappy, as they're not started yet
    when the configuration window is running.

    Args:
      getter: The method for getting the current value of the setting. If not
        given, the returned value is simply the last one that was set.
      setter: The method for setting the current value of the setting. If not
        given, the value to be set is simply stored.
    
    .. versionadded:: 2.0.0
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
    """Creates the settings needed for generating a software ROI.

    The ROI is a rectangular area defining which part of the image to keep and
    return to the :class:`~crappy.blocks.Camera` Block. It can be tuned by the
    user by setting the position of the upper left corner as well as the width
    and the height of the rectangular box.

    Using a ROI reduces the size of the image for processing, displaying and
    saving, which improves the overall performance.

    This method creates instances of
    :class:`~crappy.camera.meta_camera.camera_setting.CameraScaleSetting`
    using reserved names, and predefined choices and default. It takes as
    arguments the dimensions of the un-cropped image, so that it can adjust
    the boundaries of the sliders. These boundaries can later be adjusted
    using the :meth:`reload_software_roi` method.

    The reserved names for the instantiated settings are ``'ROI_x'``,
    ``'ROI_y'``, ``'ROI_width'``, and ``'ROI_height'``, respectively for the
    `x` and `y` position of the upper-left corner of the ROi and for the width
    and height of the ROI.

    Important:
      To apply the ROI and crop the image to return, it is necessary to call
      the :meth:`apply_soft_roi` method. Example :
      ::

        ...
        frame = self._cam.read()
        return time.time(), self.apply_soft_roi(img)

    Args:
      width: The width of the acquired images, in pixels.
      height: The height of the acquired images, in pixels.
    
    .. versionadded:: 2.0.0
    """

    # Checking that the software ROI setting does not already exist
    if self._soft_roi_set:
      raise ValueError("There can only be one set of software settings per "
                       "camera !")

    # Instantiating the CameraSetting objects
    self.log(logging.INFO, "Adding the software ROI settings")
    self.settings[self.roi_x_name] = CameraScaleSetting(
        name=self.roi_x_name, lowest=0, highest=width - 2, getter=None,
        setter=None, default=0, step=1)
    self.settings[self.roi_y_name] = CameraScaleSetting(
        name=self.roi_y_name, lowest=0, highest=height - 2, getter=None,
        setter=None, default=0, step=1)
    self.settings[self.roi_width_name] = CameraScaleSetting(
        name=self.roi_width_name, lowest=2, highest=width, getter=None,
        setter=None, default=width, step=1)
    self.settings[self.roi_height_name] = CameraScaleSetting(
        name=self.roi_height_name, lowest=2, highest=height, getter=None,
        setter=None, default=height, step=1)

    self._soft_roi_set = True

  def reload_software_roi(self, width: int, height: int) -> None:
    """Updates the software ROI boundaries when the width and/or the height of
    the acquired images change.

    The :meth:`add_software_roi` method should have been called before calling
    this method.

    Args:
      width: The width of the acquired images, in pixels.
      height: The height of the acquired images, in pixels.
    
    .. versionadded:: 2.0.0
    """

    if self._soft_roi_set:
      self.log(logging.DEBUG, "Reloading the software ROI settings")
      self.settings[self.roi_x_name].reload(lowest=0, highest=width-1,
                                            value=0, default=0)
      self.settings[self.roi_y_name].reload(lowest=0, highest=height - 1,
                                            value=0, default=0)
      self.settings[self.roi_width_name].reload(lowest=1, highest=width,
                                                value=width, default=width)
      self.settings[self.roi_height_name].reload(lowest=1, highest=height,
                                                 value=height, default=height)
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
    image. Returns the original image if the software ROI settings were not
    defined using :meth:`add_software_roi`.
    
    .. versionadded:: 2.0.0
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
    """Sets all the setting values on the camera.

    The provided keys of the kwargs should be the names of valid
    :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting`, otherwise
    an error is raised.

    For settings that are given in the kwargs, sets them to the given
    corresponding value. For the other settings, sets them to their default
    value.

    This method should be called during the :meth:`open` method, once the
    communication with the camera is established and the settings are
    instantiated. It is usually called at the very end of the method.
    """

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

    It is called in case :meth:`~crappy.camera.Camera.__getattribute__` doesn't
    work properly, and tries to return the corresponding setting value."""

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
