# coding: utf-8

from typing import Union, Literal
import logging

from .meta_inout import InOut
from .._global import OptionalModule

try:
  import RPi.GPIO as GPIO
except (ModuleNotFoundError, ImportError):
  GPIO = OptionalModule("RPi.GPIO")

try:
  import board
except (ModuleNotFoundError, ImportError):
  board = OptionalModule('board', 'Blinka is necessary to access the GPIO')

try:
  import digitalio
except (ImportError, ModuleNotFoundError):
  digitalio = OptionalModule('digitalio',
                             'Blinka is necessary to access the GPIOs')

gpio_switch_backends = ['Pi4', 'blinka']


class GPIOSwitch(InOut):
  """This class can drive a GPIO high or low on a single board computer.

  When the command value is `1` the GPIO is turned high, when the command is
  `0` it is turned low. Any value other than `0` and `1` raises an error.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Gpio_switch* to *GPIOSwitch*
  """

  def __init__(self,
               pin_out: Union[int, str],
               backend: Literal['Pi4', 'blinka']) -> None:
    """Checks the validity of the arguments.

    Args:
      pin_out: The GPIO pin to be controlled. On Raspberry Pi, should be an
        :obj:`int` corresponding to a GPIO in BCM convention. With the
        `'blinka'` backend, should be a string holding the name of the pin.
        Refer to blinka's specific documentation for each board for more
        information.
      backend: Should be one of :
        ::

          'Pi4', 'blinka'

        The `'Pi4'` backend only works on the Raspberry Pis. The `'blinka'`
        backend requires installing :mod:`Adafruit-Blinka`, but this module is
        compatible with and maintained on a wide variety of boards.

        .. versionadded:: 1.5.10
    
    .. versionadded:: 1.5.10 *ft232h_ser_num* argument
    .. versionremoved:: 2.0.0 *ft232h_ser_num* argument
    """

    self._pin_out = None

    # Checking that the backend is valid
    if not isinstance(backend, str) or backend not in gpio_switch_backends:
      raise ValueError("backend should be in {}".format(gpio_switch_backends))
    self._backend = backend

    # Checking that the given pin is valid for the selected backend
    if backend == 'blinka' and not hasattr(board, pin_out):
      raise TypeError(f'{pin_out} is not a valid pin using the blinka backend '
                      f'on this board !')
    elif backend == 'Pi4' and (not isinstance(pin_out, int) or
                               pin_out not in range(2, 28)):
      raise ValueError('pin_out should be an integer between 2 and 28 when '
                       'using the Pi4 backend !')

    super().__init__()

    # Instantiating the pin object
    if backend == 'Pi4':
      self._pin_out = pin_out
    elif backend == 'blinka':
      self._pin_out = digitalio.DigitalInOut(getattr(board, pin_out))

  def open(self) -> None:
    """Initializes the GPIO."""

    if self._backend == 'Pi4':
      self.log(logging.INFO, "Setting up the GPIOs")
      GPIO.setmode(GPIO.BCM)
      GPIO.setup(self._pin_out, GPIO.OUT)

    elif self._backend == 'blinka':
      self._pin_out.direction = digitalio.Direction.OUTPUT

  def set_cmd(self, *cmd: int) -> None:
    """Drives the GPIO according to the command.

    Args:
      cmd: 1 for driving the GPIO high, 0 for driving it low.
    """

    if cmd[0] not in [0, 1]:
      raise ValueError("The GPIO input can only be 0 or 1")

    if self._backend == 'Pi4':
      GPIO.output(self._pin_out, cmd[0])
    elif self._backend == 'blinka':
      self._pin_out.value = cmd[0]

  def close(self) -> None:
    """Releases the GPIO."""

    if self._backend == 'Pi4':
      self.log(logging.INFO, "Cleaning up the GPIOs")
      GPIO.cleanup()
    elif self._backend == 'blinka' and self._pin_out is not None:
      self._pin_out.deinit()
