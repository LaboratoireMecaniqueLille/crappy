# coding: utf-8

from __future__ import annotations
from typing import Tuple, Optional, Callable, Union
from re import findall, search, finditer, split, Match
from dataclasses import dataclass
import logging
from subprocess import run
from multiprocessing import current_process


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


class V4L2:
  """A class for getting parameters available in a camera by using v4l-utils.
  """

  def __init__(self):
    """Simply initializes the instance attributes."""

    self._parameters = list()
    self._formats = list()
    self._logger: Optional[logging.Logger] = None

  def _get_param(self, device: Optional[Union[str, int]]) -> None:
    """Extracts the different parameters and their information
     by parsing v4l2-ctl with regex."""

    # Trying to run v4l2-ctl to get the available settings
    command = ['v4l2-ctl', '-L'] if device is None \
        else ['v4l2-ctl', '-d', str(device), '-L']
    self.log(logging.INFO, f"Getting the available image settings with "
                           f"command {command}")
    try:
      check = run(command, capture_output=True, text=True)
    except FileNotFoundError:
      check = None
    check = check.stdout if check is not None else ''

    # Extract the different parameters and their information
    matches = finditer(Parameter.param_pattern, check)
    for match in matches:
      self._parameters.append(Parameter.parse_info(match))

    # Regex to extract the different options in a menu
    menu_options = finditer(Parameter.option_pattern, check)

    # Extract the different options
    for menu_option in menu_options:
      for param in self._parameters:
        param.add_options(menu_option)

  def _get_available_formats(self, device: Optional[Union[str, int]]) -> None:
    """Extracts the different formats available
    by parsing v4l2-ctl with regex."""

    # Trying to run v4l2-ctl to get the available formats
    command = ['v4l2-ctl', '--list-formats-ext'] if device is None \
        else ['v4l2-ctl', '-d', str(device), '--list-formats-ext']
    self.log(logging.INFO, f"Getting the available image formats with "
                           f"command {command}")
    try:
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
        sizes = findall(r'\d+x\d+', img_format)
        fps_sections = split(r'\d+x\d+', img_format)[1:]

        # For each name, finding the available sizes
        for size, fps_section in zip(sizes, fps_sections):
          fps_list = findall(r'\((\d+\.\d+)\sfps\)', fps_section)
          for fps in fps_list:
            self._formats.append(f'{name} {size} ({fps} fps)')

  def _add_setter(self,
                  name: str,
                  device: Optional[Union[int, str]]) -> Callable:
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
        if device is not None:
          command = ['v4l2-ctl', '-d', str(device), '--set-ctrl',
                     f'{name}={value}']
        else:
          command = ['v4l2-ctl', '--set-ctrl', f'{name}={int(value[0])}']
        self.log(logging.DEBUG, f"Setting {name} with command {command}")
        run(command, capture_output=True, text=True)
      else:
        if device is not None:
          command = ['v4l2-ctl', '-d', str(device), '--set-ctrl',
                     name+f'={int(value)}']
        else:
          command = ['v4l2-ctl', '--set-ctrl', f'{name}={int(value)}']
        self.log(logging.DEBUG, f"Setting {name} with command {command}")
        run(command, capture_output=True, text=True)
    return setter

  @staticmethod
  def _add_scale_getter(name: str,
                        device: Optional[Union[int, str]]) -> Callable:
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
      if device is not None:
        command = ['v4l2-ctl', '-d', str(device), '--get-ctrl', name]
      else:
        command = ['v4l2-ctl', '--get-ctrl', name]
      try:
        value = run(command, capture_output=True, text=True).stdout
        value = search(r': (-?\d+)', value).group(1)
      except FileNotFoundError:
        value = None
      return int(value)
    return getter

  @staticmethod
  def _add_bool_getter(name: str,
                       device: Optional[Union[int, str]]) -> Callable:
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
      if device is not None:
        command = ['v4l2-ctl', '-d', str(device), '--get-ctrl', name]
      else:
        command = ['v4l2-ctl', '--get-ctrl', name]
      try:
        value = run(command, capture_output=True, text=True).stdout
        value = search(r': (\d+)', value).group(1)
      except FileNotFoundError:
        value = None
      return bool(int(value))
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
      running v4l2-ctl.
      """

      # Trying to run v4l2-ctl to get the value
      if device is not None:
        command = ['v4l2-ctl', '-d', str(device), '--get-ctrl', name]
      else:
        command = ['v4l2-ctl', '--get-ctrl', name]
      try:
        value = run(command, capture_output=True, text=True).stdout
        value = search(r': (\d+)', value).group(1)
        for param in self._parameters:
          if param.name == name:
            for option in param.options:
              if value == search(r'(\d+):', option).group(1):
                value = option
      except FileNotFoundError:
        value = None
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
