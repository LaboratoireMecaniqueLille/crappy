# coding: utf-8

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
               getter: Callable[[], str] | None = None,
               setter: Callable[[str], None] | None = None,
               default: str | None = None) -> None:
    """Sets the attributes.

    Args:
      name: The name of the setting, that will be displayed in the GUI.
      choices: A tuple listing the possible values for the setting.
      getter: The method for getting the current value of the setting.
      setter: The method for setting the current value of the setting.
      default: The default value to assign to the setting.
    """

    self.choices = choices

    if default is not None and default not in choices:
      self.log(logging.WARNING, f"The given default {default} is not part "
                                  f"of the given choices ! Setting default "
                                  f"to {choices[0]} instead")
      default = choices[0]

    if default is None:
      default = choices[0]

    super().__init__(name, getter, setter, default)

    self.tk_obj = list()

  def reload(self,
             choices: tuple[str, ...],
             value: str | None = None,
             default: str | None = None) -> None:
    """Allows modifying the choices of the radio buttons once they have been
    instantiated.

    Note:
      As the layout of the GUI is already fixed, the number of displayed
      options cannot vary. It is thus not possible to propose more choices than
      those initially proposed. Reversely, if fewer new options ar proposed
      then some radio buttons will be disabled.

    Args:
      choices: The new possible choices for the setting.
      value: Optionally, a value to set the setting to when reloading it. If
        not provided, the current value remains if it is in the new choices,
        otherwise the default is set.

        .. versionadded:: 2.0.7
      default: Optionally, the new default value for the setting. If not
        provided, the previous default remains, if it is still in the new
        choices. Otherwise, a new default is defined.

    .. versionadded:: 2.0.0
    """

    self.log(logging.DEBUG, f"Reloading the setting {self.name}")

    # Updating the default value
    if default is not None:
      if default in choices:
        self.default = default
      else:
        self.log(logging.WARNING, f"The given default {default} is not part "
                                  f"of the given choices ! Setting default "
                                  f"to {choices[0]} instead")
        self.default = choices[0]
    else:
      self.default = choices[0]

    if value is not None and value not in choices:
      self.log(logging.WARNING, f"{value} is not a possible choice for the "
                                f"setting {self.name}, ignoring it !")
      value  = None

    if value is None and self.value not in choices:
      self.log(logging.WARNING, f"{self.value} is no longer a possible choice "
                                f"for the setting {self.name}, setting to "
                                f"{self.default} instead !")
      value = self.default

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
          continue

        # Updating the text and value of the button, and enabling it
        button.configure(value=choice, text=choice, state='normal')

    if value is not None:
      if self.tk_var is None:
        # If the setting was never set, not setting it yet but tweaking its
        # default so that it will only be set to the right value when expected
        if not self.was_set:
          self.log(logging.DEBUG, f"Setting default to {value} as a hack to "
                                  f"maintain settings call order")
          self.default = value
        # If the setting was already set though a kwarg, the user probably
        # doesn't want it silently overridden by a reload() call
        elif self.user_set and value != self.value:
          raise ValueError(f"Setting {self.name} forcibly set to value "
                           f"{self.value} using a kwarg, cannot override to "
                           f"{value} !")
        # Setting to the indicated value if parameter was previously set to
        # default
        else:
          self.value = value
      # Once in the graphical interface it is assumed that the user does not
      # want strict control over settings, always setting
      else:
        self.value = value
