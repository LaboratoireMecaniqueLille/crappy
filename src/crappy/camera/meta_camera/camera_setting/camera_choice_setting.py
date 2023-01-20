# coding: utf-8

from typing import Optional, Callable, Tuple

from .camera_setting import CameraSetting


class CameraChoiceSetting(CameraSetting):
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
