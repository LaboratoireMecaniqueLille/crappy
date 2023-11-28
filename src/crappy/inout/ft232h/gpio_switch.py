# coding: utf-8

from typing import Union

from ..meta_inout import InOut
from ...tool.ft232h import FT232HServer as FT232H, ft232h_pin_nr, USBArgsType


class GPIOSwitchFT232H(InOut):
  """This class can drive a GPIO high or low on a FT232H.

  It is similar to the :class:`~crappy.inout.GPIOSwitch` class, except this
  class is specific for use with an :class:`~crappy.tool.ft232h.FT232H` USB to
  GPIO converter.

  When the command value is `1` the GPIO is turned high, when the command is
  `0` it is turned low. Any value other than `0` and `1` raises an error.
  
  .. versionadded:: 2.0.0
  """

  ft232h = True

  def __init__(self,
               pin_out: Union[int, str],
               _ft232h_args: USBArgsType = tuple()) -> None:
    """Checks the validity of the arguments.

    Args:
      pin_out: The GPIO pin to be controlled. On Raspberry Pi, should be an
        integer corresponding to a GPIO in BCM convention. On FT232H, should be
        a string corresponding to the name of a GPIO. With the `'blinka'`
        backend, should be a string holding the name of the pin. Refer to
        blinka's specific documentation for each board for more information.
      _ft232h_args: This argument is meant for internal use only and should not
        be provided by the user. It contains the information necessary for
        setting up the FT232H.
    """

    self._ft232h = None
    self._pin_out = None

    # Checking that the given pin is valid
    if pin_out not in ft232h_pin_nr:
      raise TypeError(f'{pin_out} is not a valid pin for the ft232h backend !')

    super().__init__()

    (block_index, block_lock, command_file, answer_file, shared_lock,
     current_block) = _ft232h_args

    self._ft232h = FT232H(mode='GPIO_only',
                          block_index=block_index,
                          current_block=current_block,
                          command_file=command_file,
                          answer_file=answer_file,
                          block_lock=block_lock,
                          shared_lock=shared_lock)

    self._pin_out = pin_out

  def set_cmd(self, *cmd: int) -> None:
    """Drives the GPIO according to the command.

    Args:
      cmd: 1 for driving the GPIO high, 0 for driving it low.
    """

    if cmd[0] not in [0, 1]:
      raise ValueError("The GPIO input can only be 0 or 1")

    self._ft232h.set_gpio(self._pin_out, cmd[0])

  def close(self) -> None:
    """Releases the GPIO."""

    if self._ft232h is not None:
      self._ft232h.close()
