# coding: utf-8

from typing import Optional
from collections.abc import Callable
from itertools import zip_longest
import logging

from .camera_setting import CameraSetting


class CameraChoiceSetting(CameraSetting):
  """Camera setting that can take any value from a predefined list of
  values.

  It is a child of
  :class:`~crappy.camera.meta_camera.camera_setting.CameraSetting`.
  
  .. versionadded:: 1.5.10
  .. versionchanged:: 2.0.0
     renamed from *Camera_choice_setting* to *CameraChoiceSetting*
  """

  def __init__(self,
               name: str,
               choices: tuple[str, ...],
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

    self.tk_obj = list()

  def reload(self,
             choices: tuple[str, ...],
             default: Optional[str] = None) -> None:
    """Allows modifying the choices of the radio buttons once they have been
    instantiated.

    As the layout of the GUI is already fixed, the number of displayed options
    cannot vary. It is thus not possible to propose more choices than those
    initially proposed. Reversely, if fewer new options ar proposed then some
    radio buttons won't be affected a value.
    
    .. versionadded:: 2.0.0
    """

    self.log(logging.DEBUG, f"Reloading the setting {self.name}")

    # Updating the default value
    if default is not None:
      self.default = default
    else:
      self.default = choices[0]

    # Updating the radio buttons and the setting value
    if self.tk_obj:
      for button, choice in zip_longest(self.tk_obj, choices):
        # If there are more choices than buttons, ignoring the extra choices
        if button is None:
          self.log(logging.WARNING,
                   f"Too many choices given when reloading the {self.name} "
                   f"setting, ignoring the extra ones")
          break
        # If there are more buttons than choices, disabling the extra buttons
        if choice is None:
          self.log(logging.WARNING,
                   f"Too few choices given when reloading the {self.name} "
                   f"setting, disabling the extra buttons")
          button.configure(state='disabled', value='', text='')

        # Updating the text and value of the button, and enabling it
        button.configure(value=choice, text=choice, state='normal')
    if self.tk_var is not None:
      self.tk_var.set(self.value)
