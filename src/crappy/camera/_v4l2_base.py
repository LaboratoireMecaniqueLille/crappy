# coding: utf-8

from __future__ import annotations
from typing import Optional, Union
from collections.abc import Callable
from re import findall, search, finditer, split, Match, compile
from dataclasses import dataclass
import logging
from subprocess import run
from multiprocessing import current_process


@dataclass
class V4L2Parameter:
  """A class for the different parameters the user can adjust."""

  name: str
  type: str
  min: Optional[str] = None
  max: Optional[str] = None
  step: Optional[str] = None
  default: Optional[str] = None
  value: Optional[str] = None
  flags: Optional[str] = None
  options: Optional[tuple[str, ...]] = None

  # Regex to extract the different parameters and their information
  param_pattern = (r'(\w+)\s+0x\w+\s+\((\w+)\)\s+:\s*'
                   r'(min=(-?\d+)\s+)?'
                   r'(max=(-?\d+)\s+)?'
                   r'(step=(\d+)\s+)?'
                   r'(default=(-?\d+)\s+)?'
                   r'value=(-?\d+)\s*'
                   r'(flags=([^\\n]+))?')

  option_pattern = r'(\w+ \w+ \(menu\))([\s\S]+?)(?=\n\s*\w+ \w+ \(.+?\)|$)'

  @classmethod
  def parse_info(cls, match: Match) -> V4L2Parameter:
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
      for num, opt in zip(num_options, options):
        if self.default == num:
          self.default = opt


class V4L2Helper:
  """A class for getting parameters available in a camera by using
  v4l-utils.

  .. versionadded:: 2.0.0
  """

  def __init__(self):
    """Simply initializes the instance attributes."""

    self._parameters = list()
    self._formats = list()
    self._logger: Optional[logging.Logger] = None

  def _get_param(self, device: Optional[Union[str, int]]) -> None:
    """Extracts the different parameters and their information by parsing
    v4l2-ctl with regex."""

    # Trying to run v4l2-ctl to get the available settings
    if device is None:
      command = ['v4l2-ctl', '-L']
    else:
      command = ['v4l2-ctl', '-d', str(device), '-L']
    self.log(logging.DEBUG, f"Getting the available image settings with "
                            f"command {' '.join(command)}")
    ret = run(command, capture_output=True, text=True).stdout
    self.log(logging.DEBUG, f"Got the following image settings: {ret}")

    # Extract the different parameters and their information
    matches = finditer(V4L2Parameter.param_pattern, ret)
    for match in matches:
      self._parameters.append(V4L2Parameter.parse_info(match))

    # Regex to extract the different options in a menu
    menu_options = finditer(V4L2Parameter.option_pattern, ret)

    # Extract the different options
    for menu_option in menu_options:
      for param in self._parameters:
        param.add_options(menu_option)

  @staticmethod
  def _sort_key(format_: str):
    """Key function to sort the different formats."""

    format_pattern = compile(r'(.+?)\s+(\d+)x(\d+)\s+\((\d+\.\d+) fps\)')
    match = format_pattern.search(format_)
    name = match.group(1)
    width = int(match.group(2))
    fps = float(match.group(4))
    return name, fps, width

  def _get_available_formats(self, device: Optional[Union[str, int]]) -> None:
    """Extracts the different formats available by parsing v4l2-ctl with
    regex."""

    # Trying to run v4l2-ctl to get the available formats
    if device is None:
      command = ['v4l2-ctl', '--list-formats-ext']
    else:
      command = ['v4l2-ctl', '-d', str(device), '--list-formats-ext']
    self.log(logging.DEBUG, f"Getting the available image formats with "
                            f"command {' '.join(command)}")
    ret = run(command, capture_output=True, text=True).stdout
    self.log(logging.DEBUG, f"Got the following image formats: {ret}")

    # Splitting the returned string to isolate each encoding
    if findall(r'\[\d+]', ret):
      formats = split(r'\[\d+]', ret)[1:]
    elif findall(r'Pixel\sFormat', ret):
      formats = split(r'Pixel\sFormat', ret)[1:]
    else:
      formats = list()

    # For each encoding, finding its name, available sizes and framerates
    if formats:
      for img_format in formats:
        name, *_ = search(r"'(\w+)'", img_format).groups()
        sizes = findall(r'\d+x\d+', img_format)
        fps_sections = split(r'\d+x\d+', img_format)[1:]

        # Formatting the detected sizes and framerates into strings
        for size, fps_section in zip(sizes, fps_sections):
          fps_list = findall(r'\((\d+\.\d+)\sfps\)', fps_section)
          for fps in fps_list:
            self._formats.append(f'{name} {size} ({fps} fps)')
      self._formats = sorted(set(self._formats), key=self._sort_key)

  def _add_setter(self,
                  name: str,
                  device: Optional[Union[int, str]]) -> Callable:
    """Creates a setter function for a setting named 'name'.

    Args:
      name: Name of the setting.

    Returns:
      The setter function.
    """

    def setter(value: Union[str, int, bool]) -> None:
      """The method to set the value of a setting running v4l2-ctl."""

      if isinstance(value, str):
        # The value to set the menu parameter is just the int
        # at the beginning the string
        value = search(r'(\d+): ', value).group(1)
        if device is not None:
          command = ['v4l2-ctl', '-d', str(device), '--set-ctrl',
                     f'{name}={value}']
        else:
          command = ['v4l2-ctl', '--set-ctrl', f'{name}={int(value[0])}']
        self.log(logging.DEBUG, f"Running the command {' '.join(command)}")
        run(command, capture_output=True, text=True)
        self.log(logging.DEBUG, f"Set {name} to {int(value[0])}")
      else:
        if device is not None:
          command = ['v4l2-ctl', '-d', str(device), '--set-ctrl',
                     f'{name}={int(value)}']
        else:
          command = ['v4l2-ctl', '--set-ctrl', f'{name}={int(value)}']
        self.log(logging.DEBUG, f"Running the command {' '.join(command)}")
        run(command, capture_output=True, text=True)
        self.log(logging.DEBUG, f"Set {name} to {int(value)}")
    return setter

  def _add_scale_getter(self,
                        name: str,
                        device: Optional[Union[int, str]]) -> Callable:
    """Creates a getter function for a setting named 'name'.

    Args:
      name: Name of the setting.

    Returns:
      The getter function.
    """

    def getter() -> int:
      """The method to get the current value of a scale setting running
       v4l2-ctl."""

      # Trying to run v4l2-ctl to get the value
      if device is not None:
        command = ['v4l2-ctl', '-d', str(device), '--get-ctrl', name]
      else:
        command = ['v4l2-ctl', '--get-ctrl', name]
      self.log(logging.DEBUG, f"Running the command {' '.join(command)}")
      value = run(command, capture_output=True, text=True).stdout
      value = int(search(r':\s(-?\d+)', value).group(1))
      self.log(logging.DEBUG, f"Got {name}: {value}")
      return value
    return getter

  def _add_bool_getter(self,
                       name: str,
                       device: Optional[Union[int, str]]) -> Callable:
    """Creates a getter function for a setting named 'name'.

    Args:
      name: Name of the setting.

    Returns:
      The getter function.
    """

    def getter() -> bool:
      """The method to get the current value of a bool setting running
      v4l2-ctl."""

      # Trying to run v4l2-ctl to get the value
      if device is not None:
        command = ['v4l2-ctl', '-d', str(device), '--get-ctrl', name]
      else:
        command = ['v4l2-ctl', '--get-ctrl', name]
      self.log(logging.DEBUG, f"Running the command {' '.join(command)}")
      value = run(command, capture_output=True, text=True).stdout
      value = bool(int(search(r':\s(\d+)', value).group(1)))
      self.log(logging.DEBUG, f"Got {name}: {value}")
      return value
    return getter

  def _add_menu_getter(self,
                       name: str,
                       device: Optional[Union[int, str]]) -> Callable:
    """Creates a getter function for a setting named 'name'.

    Args:
      name: Name of the setting.

    Returns:
      The getter function.
    """

    def getter() -> str:
      """The method to get the current value of a choice setting
      running v4l2-ctl."""

      # Trying to run v4l2-ctl to get the value
      if device is not None:
        command = ['v4l2-ctl', '-d', str(device), '--get-ctrl', name]
      else:
        command = ['v4l2-ctl', '--get-ctrl', name]
      self.log(logging.DEBUG, f"Running the command {' '.join(command)}")
      value = run(command, capture_output=True, text=True).stdout
      value = search(r':\s(\d+)', value).group(1)
      for param in self._parameters:
        if param.name == name:
          for option in param.options:
            if value == search(r'(\d+):', option).group(1):
              value = option
      self.log(logging.DEBUG, f"Got {name}: {value}")
      return value
    return getter

  def log(self, level: int, msg: str) -> None:
    """Records log messages for the Modifiers.

    Also instantiates the logger when logging the first message.

    Args:
      level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.
    """

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)
