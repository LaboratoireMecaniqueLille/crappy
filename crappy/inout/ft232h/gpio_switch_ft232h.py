# coding: utf-8

from typing import Optional, Union

from ..inout import InOut
from ...tool import ft232h_server as ft232h, Usb_server, ft232h_pin_nr


class Gpio_switch_ft232h(Usb_server, InOut):
  """Class for setting a GPIO high or low.

  The Gpio_switch InOut block is meant for switching a GPIO high or low
  according to the input signal value. When the input signal is `1` the
  GPIO is turned high, when the signal is `0` it is turned low. Any value other
  than `0` and `1` raises an error.
  """

  ft232h = True

  def __init__(self,
               pin_out: Union[int, str],
               ft232h_ser_num: Optional[int] = None) -> None:
    """Checks the argument validity.

    Args:
      pin_out: The GPIO pin to be controlled. On Raspberry Pi, should be an
        integer corresponding to a GPIO in BCM convention. On FT232H, should be
        a string corresponding to the name of a GPIO. With the `'blinka'`
        backend, should be a string holding the name of the pin. Refer to
        blinka's specific documentation for each board for more information.
      ft232h_ser_num: If backend is `'ft232h'`, the serial number of the FT232H
        to use for communication.
    """

    self._ft232h = None
    self._pin_out = None

    # Checking that the given pin is valid
    if pin_out not in ft232h_pin_nr:
      raise TypeError(f'{pin_out} is not a valid pin for the ft232h backend !')

    # Starting the USB server (doesn't do anything if backend is not ft232h)
    Usb_server.__init__(self,
                        serial_nr=ft232h_ser_num if ft232h_ser_num else '',
                        backend='ft232h')
    InOut.__init__(self)
    current_file, block_number, command_file, answer_file, block_lock, \
        current_lock = super().start_server()

    # Instantiating the pin object
    self._pin_out = pin_out
    self._ft232h = ft232h(mode='GPIO_only',
                          block_number=block_number,
                          current_file=current_file,
                          command_file=command_file,
                          answer_file=answer_file,
                          block_lock=block_lock,
                          current_lock=current_lock,
                          serial_nr=ft232h_ser_num)

  def set_cmd(self, *cmd: int) -> None:
    """Drives the GPIO according to the command.

    Args:
      cmd (:obj:`int`): 1 for driving the GPIO high, 0 for driving it low
    """

    if cmd[0] not in [0, 1]:
      raise ValueError("The GPIO input can only be 0 or 1")

    self._ft232h.set_gpio(self._pin_out, cmd[0])

  def close(self) -> None:
    """Releases the GPIO."""

    if self._ft232h is not None:
      self._ft232h.close()
