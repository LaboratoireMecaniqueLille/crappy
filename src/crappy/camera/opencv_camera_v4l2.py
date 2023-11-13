# coding: utf-8

from __future__ import annotations
from time import time, sleep
from typing import Tuple, List, Optional, Callable
from numpy import ndarray
from subprocess import run
from re import findall, split, search, finditer, Match
import logging
from dataclasses import dataclass

from .meta_camera import Camera
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


@dataclass
class Parameter:
  """A class for the different parameters the user can adjust."""

  name: str
  type: str
  min: Optional[str] = None
  max: Optional[str] = None
  step: Optional[str] = None
  default: Optional[str] = None
  value: Optional[str] = None
  flags: Optional[str] = None
  options: Optional[Tuple[str, ...]] = None

  @classmethod
  def parse_info(cls, match: Match) -> Parameter:
    """Instantiates the class Parameter, according to the information
     collected with v4l2-ctl.

    Args:
      match: Match object returned by successful matches of the regex with
      a string.

    Returns:
      The instantiated class.
    """

    return cls(name=match.group(1),
               type=match.group(2),
               min=match.group(4) if match.group(4) else None,
               max=match.group(6) if match.group(6) else None,
               step=match.group(8) if match.group(8) else None,
               default=match.group(10) if match.group(10) else None,
               value=match.group(11),
               flags=match.group(13) if match.group(13) else None)

  def add_options(self, match: Match) -> None:
    """Adds the different possible options for a menu parameter.

    Args:
      match: Match object returned by successful matches of the regex with
      a string.
    """

    menu_info = match.group(1)
    menu_values = match.group(2)
    menu_name = search(r'(\w+) \w+ \(menu\)', menu_info).group(1)
    if self.name == menu_name:
      options = findall(r'\d+: .+?(?=\n|$)', menu_values)
      num_options = findall(r'(\d+): .+?(?=\n|$)', menu_values)
      self.options = tuple(options)
      for i in range(len(num_options)):
        if self.default == num_options[i]:
          self.default = options[i]


class CameraOpencv(Camera):
  """A class for reading images from any camera able to interface with OpenCv.

  The number of the video device to read images from can be specified. It is
  then also possible to tune the encoding format and the size.

  This camera class is less performant than the
  :class:`~crappy.camera.CameraGstreamer` one that relies on GStreamer, but the
  installation of OpenCv is way easier than the one of GStreamer.

  To use this class, `v4l-utils` must be installed.
  """

  def __init__(self) -> None:
    """Sets variables and adds the channels setting."""

    super().__init__()

    self._cap = None
    self._device_num: Optional[int] = None
    self._formats: List[str] = list()
    self.parameters = []

    self.add_choice_setting(name="channels",
                            choices=('1', '3'),
                            default='1')

  def open(self, device_num: int = 0, **kwargs) -> None:
    """Opens the video stream and sets any user-specified settings.

    Args:
      device_num: The index of the device to open, as an :obj:`int`.
      **kwargs: Any additional setting to set before opening the configuration
        window.
    """

    if not isinstance(device_num, int) or device_num < 0:
      raise ValueError("device_num should be an integer !")

    # Opening the videocapture device
    self.log(logging.INFO, "Opening the image stream from the camera")
    self._cap = cv2.VideoCapture(device_num)
    self._device_num = device_num

    self._formats = []

    # Trying to run v4l2-ctl to get the available formats
    command = ['v4l2-ctl', '-d', str(device_num), '--list-formats-ext']
    try:
      self.log(logging.INFO, f"Getting the available image formats with "
                             f"command {command}")
      check = run(command, capture_output=True, text=True)
    except FileNotFoundError:
      check = None
    check = check.stdout if check is not None else ''

    # Splitting the returned string to isolate each encoding
    if findall(r'\[\d+]', check):
      check = split(r'\[\d+]', check)[1:]
    elif findall(r'Pixel\sFormat', check):
      check = split(r'Pixel\sFormat', check)[1:]
    else:
      check = []

    if check:
      for img_format in check:
        # For each encoding, finding its name
        name, *_ = search(r"'(\w+)'", img_format).groups()
        if name == 'MJPG' or name == 'YUYV':
          sizes = findall(r'\d+x\d+', img_format)
          fps_sections = split(r'\d+x\d+', img_format)[1:]

          # For each name, finding the available sizes
          for size, fps_section in zip(sizes, fps_sections):
            fps_list = findall(r'\((\d+\.\d+)\sfps\)', fps_section)
            for fps in fps_list:
              self._formats.append(f'{name} {size} ({fps} fps)')

    if self._formats:
      # The format integrates the size selection
      if ' ' in self._formats[0]:
        self.add_choice_setting(name='format',
                                choices=tuple(self._formats),
                                getter=self._get_format_size,
                                setter=self._set_format)
      # The size is independent of the format
      else:
        self.add_choice_setting(name='format',
                                choices=tuple(self._formats),
                                getter=self._get_fourcc,
                                setter=self._set_format)

    # Trying to run v4l2-ctl to get the available settings
    command = ['v4l2-ctl', '-L'] if device_num is None \
        else ['v4l2-ctl', '-d', str(device_num), '-L']
    self.log(logging.INFO, f"Getting the available image settings with "
                           f"command {command}")
    try:
      check = run(command, capture_output=True, text=True)
    except FileNotFoundError:
      check = None
    check = check.stdout if check is not None else ''

    # Regex to extract the different parameters and their information
    param_pattern = (r'(\w+)\s+0x\w+\s+\((\w+)\)\s+:\s*'
                     r'(min=(-?\d+)\s+)?'
                     r'(max=(-?\d+)\s+)?'
                     r'(step=(\d+)\s+)?'
                     r'(default=(-?\d+)\s+)?'
                     r'value=(-?\d+)\s*'
                     r'(flags=([^\\n]+))?')

    # Extract the different parameters and their information
    matches = finditer(param_pattern, check)
    for match in matches:
      self.parameters.append(Parameter.parse_info(match))

    # Regex to extract the different options in a menu
    menu_options = finditer(
      r'(\w+ \w+ \(menu\))([\s\S]+?)(?=\n\s*\w+ \w+ \(.+?\)|$)', check)

    # Extract the different options
    for menu_option in menu_options:
      for param in self.parameters:
        param.add_options(menu_option)

    # Create the different settings
    for param in self.parameters:
      if not param.flags:
        if param.type == 'int':
          self.add_scale_setting(name=param.name,
                                 lowest=int(param.min),
                                 highest=int(param.max),
                                 getter=self._add_scale_getter(param.name),
                                 setter=self._add_setter(param.name),
                                 default=param.default,
                                 step=int(param.step))
        elif param.type == 'bool':
          self.add_bool_setting(name=param.name,
                                getter=self._add_bool_getter(param.name),
                                setter=self._add_setter(param.name),
                                default=bool(int(param.default)))
        elif param.type == 'menu':
          if param.options:
            self.add_choice_setting(name=param.name,
                                    choices=param.options,
                                    getter=self._add_menu_getter(param.name),
                                    setter=self._add_setter(param.name),
                                    default=param.default)
        else:
          self.log(logging.ERROR, f'The type {param.type} is not yet'
                                  f' implemented. Only int, bool and menu '
                                  f'type are implemented. ')
          raise NotImplementedError

    # Adding the software ROI selection settings
    if 'width' in self.settings and 'height' in self.settings:
      width, height = self._get_width(), self._get_height()
      self.add_software_roi(width, height)
    elif 'format' in self.settings:
      width, height = search(r'(\d+)x(\d+)', self._get_format_size()).groups()
      self.add_software_roi(int(width), int(height))

    # Setting the kwargs if any
    self.set_all(**kwargs)

  def get_image(self) -> Tuple[float, ndarray]:
    """Grabs a frame from the videocapture object and returns it along with a
    timestamp."""

    # Grabbing the frame and the timestamp
    t = time()
    ret, frame = self._cap.read()

    # Checking the integrity of the frame
    if not ret:
      raise IOError("Error reading the camera")

    # Returning the image in the right format, and its timestamp
    if self.channels == '1':
      return t, self.apply_soft_roi(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    else:
      return t, self.apply_soft_roi(frame)

  def close(self) -> None:
    """Releases the videocapture object."""

    if self._cap is not None:
      self.log(logging.INFO, "Closing the image stream from the camera")
      self._cap.release()

  def _get_fourcc(self) -> str:
    """Returns the current fourcc string of the video encoding."""

    fcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
    return f"{fcc & 0xFF:c}{(fcc >> 8) & 0xFF:c}" \
           f"{(fcc >> 16) & 0xFF:c}{(fcc >> 24) & 0xFF:c}"

  def _get_width(self) -> int:
    """Returns the current image width."""

    return self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)

  def _get_height(self) -> int:
    """Returns the current image height."""

    return self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

  def _set_width(self, width: int) -> None:
    """Tries to set the image width."""

    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    # Reloading the software ROI selection settings
    if self._soft_roi_set:
      sleep(0.1)
      width, height = self._get_width(), self._get_height()
      self.reload_software_roi(width, height)

  def _set_height(self, height: int) -> None:
    """Tries to set the image height."""

    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Reloading the software ROI selection settings
    if self._soft_roi_set:
      sleep(0.1)
      width, height = self._get_width(), self._get_height()
      self.reload_software_roi(width, height)

  def _set_format(self, img_format: str) -> None:
    """Sets the format of the image according to the user's choice."""

    # The format might be made of a name and a dimension, or just a name
    try:
      format_name, img_size, fps = findall(r"(\w+)\s(\w+)\s\((\d+.\d+) fps\)",
                                           img_format)[0]
    except ValueError:
      format_name, img_size, fps = img_format, None, None

    # Setting the format
    self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*format_name))

    if img_size is not None:
      # Getting the width and height from the second half of the string
      width, height = map(int, img_size.split('x'))

      # Setting the size
      self._set_width(width)
      self._set_height(height)

    if fps is not None:
      self._cap.set(cv2.CAP_PROP_FPS, float(fps))

    # Reloading the software ROI selection settings
    if self._soft_roi_set:
      sleep(0.1)
      width, height = search(r'(\d+)x(\d+)', self._get_format_size()).groups()
      self.reload_software_roi(int(width), int(height))

  def _get_format_size(self) -> str:
    """Parses the ``v4l2-ctl -V`` command to get the current image format as a
    :obj:`str`."""

    # Sending the v4l2-ctl command
    command = ['v4l2-ctl', '-d', str(self._device_num), '--all']
    check = run(command, capture_output=True, text=True).stdout

    # Parsing the answer
    format_ = width = height = fps = ''
    if search(r"Pixel Format\s*:\s*'(\w+)'", check) is not None:
      format_, *_ = search(r"Pixel Format\s*:\s*'(\w+)'", check).groups()
    if search(r"Width/Height\s*:\s*(\d+)/(\d+)", check) is not None:
      width, height = search(r"Width/Height\s*:\s*(\d+)/(\d+)", check).groups()
    if search(r"Frames per second\s*:\s*(\d+.\d+)", check) is not None:
      fps, *_ = search(r"Frames per second\s*:\s*(\d+.\d+)", check).groups()

    return f'{format_} {width}x{height} ({fps} fps)'

  def _add_setter(self, name: str) -> Callable:
    """Creates a setter function for a setting named 'name'.
    Args:
      name: Name of the setting.

    Returns:
      The setter function.
    """

    def setter(value) -> None:
      """The method to set the value of a setting running v4l2-ctl.
      """

      if isinstance(value, str):
        # The value to set the menu parameter is just the int
        # at the beginning the string
        value = search(r'(\d+): ', value).group(1)
        if self._device_num is not None:
          command = ['v4l2-ctl', '-d', str(self._device_num), '--set-ctrl',
                     f'{name}={value}']
        else:
          command = ['v4l2-ctl', '--set-ctrl', f'{name}={int(value[0])}']
        self.log(logging.DEBUG, f"Setting {name} with command {command}")
        run(command, capture_output=True, text=True)
      else:
        if self._device_num is not None:
          command = ['v4l2-ctl', '-d', str(self._device_num), '--set-ctrl',
                     name+f'={int(value)}']
        else:
          command = ['v4l2-ctl', '--set-ctrl', f'{name}={int(value)}']
        self.log(logging.DEBUG, f"Setting {name} with command {command}")
        run(command, capture_output=True, text=True)
    return setter

  def _add_scale_getter(self, name: str) -> Callable:
    """Creates a getter function for a setting named 'name'.
    Args:
      name: Name of the setting.

    Returns:
      The getter function.
    """

    def getter() -> int:
      """The method to get the current value of a scale setting
      running v4l2-ctl.
      """

      # Trying to run v4l2-ctl to get the value
      if self._device_num is not None:
        command = ['v4l2-ctl', '-d', str(self._device_num), '--get-ctrl', name]
      else:
        command = ['v4l2-ctl', '--get-ctrl', name]
      try:
        self.log(logging.DEBUG, f"Getting {name} with command {command}")
        value = run(command, capture_output=True, text=True).stdout
        value = search(r': (-?\d+)', value).group(1)
      except FileNotFoundError:
        value = None
      return int(value)
    return getter

  def _add_bool_getter(self, name: str) -> Callable:
    """Creates a getter function for a setting named 'name'.
    Args:
      name: Name of the setting.

    Returns:
      The getter function.
    """

    def getter() -> bool:
      """The method to get the current value of a bool setting
      running v4l2-ctl.
      """

      # Trying to run v4l2-ctl to get the value
      if self._device_num is not None:
        command = ['v4l2-ctl', '-d', str(self._device_num), '--get-ctrl', name]
      else:
        command = ['v4l2-ctl', '--get-ctrl', name]
      try:
        self.log(logging.DEBUG, f"Getting {name} with command {command}")
        value = run(command, capture_output=True, text=True).stdout
        value = search(r': (\d+)', value).group(1)
      except FileNotFoundError:
        value = None
      return bool(int(value))
    return getter

  def _add_menu_getter(self, name: str) -> Callable:
    """Creates a getter function for a setting named 'name'.
    Args:
      name: Name of the setting.

    Returns:
      The getter function.
    """

    def getter() -> str:
      """The method to get the current value of a choice setting
      running v4l2-ctl.
      """

      # Trying to run v4l2-ctl to get the value
      if self._device_num is not None:
        command = ['v4l2-ctl', '-d', str(self._device_num), '--get-ctrl', name]
      else:
        command = ['v4l2-ctl', '--get-ctrl', name]
      try:
        self.log(logging.DEBUG, f"Getting {name} with command {command}")
        value = run(command, capture_output=True, text=True).stdout
        value = search(r': (\d+)', value).group(1)
        for param in self.parameters:
          if param.name == name:
            for option in param.options:
              if value == search(r'(\d+):', option).group(1):
                value = option
      except FileNotFoundError:
        value = None
      return value
    return getter
