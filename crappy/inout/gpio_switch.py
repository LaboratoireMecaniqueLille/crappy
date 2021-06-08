# coding: utf-8

from .inout import InOut
from .._global import OptionalModule

try:
  import RPi.GPIO as GPIO
except (ModuleNotFoundError, ImportError):
  GPIO = OptionalModule("RPi.GPIO")


class Gpio_switch(InOut):
  """Class for setting a GPIO high or low.

  The Gpio_switch InOut block is meant for switching a Raspberry Pi GPIO high
  or low according to the input signal value. When the input signal is `1` the
  GPIO is turned high, when the signal is `0` it is turned low. Any value other
  than `0` and `1` raises an error.

  Warning:
    Only works on Raspberry Pi !
  """

  def __init__(self, pin_out: int) -> None:
    """Checks the argument validity.

    Args:
      pin_out (:obj:`int`): The GPIO pin to be controlled (BCM convention).
    """

    InOut.__init__(self)
    if pin_out not in range(2, 28):
      raise ValueError("pin_out should be an integer between 2 and 28")
    self._pin_out = pin_out

  def open(self) -> None:
    """Sets the GPIO."""

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(self._pin_out, GPIO.OUT)

  def set_cmd(self, *cmd: int) -> None:
    """Drives the GPIO according to the command.

    Args:
      cmd (:obj:`int`): 1 for driving the GPIO high, 0 for driving it low
    """

    if cmd[0] not in [0, 1]:
      raise ValueError("The GPIO input can only be 0 or 1")
    else:
      GPIO.output(self._pin_out, cmd[0])

  @staticmethod
  def close() -> None:
    """Releases the GPIO."""

    GPIO.cleanup()
