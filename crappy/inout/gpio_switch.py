# coding: utf-8

from typing import Optional, Union, Literal
from .inout import InOut
from .._global import OptionalModule
from ..tool import ft232h_server as ft232h, Usb_server, ft232h_pin_nr

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

gpio_switch_backends = ['Pi4', 'ft232h', 'blinka']


class Gpio_switch(Usb_server, InOut):
  """Class for setting a GPIO high or low.

  The Gpio_switch InOut block is meant for switching a GPIO high or low
  according to the input signal value. When the input signal is `1` the
  GPIO is turned high, when the signal is `0` it is turned low. Any value other
  than `0` and `1` raises an error.
  """

  def __init__(self,
               pin_out: Union[int, str],
               backend: Literal['Pi4', 'ft232h', 'blinka'],
               ft232h_ser_num: Optional[int] = None) -> None:
    """Checks the argument validity.

    Args:
      pin_out: The GPIO pin to be controlled. On Raspberry Pi, should be an
        integer corresponding to a GPIO in BCM convention. On FT232H, should be
        a string corresponding to the name of a GPIO. With the `'blinka'`
        backend, should be a string holding the name of the pin. Refer to
        blinka's specific documentation for each board for more information.
      backend: Should be one of :
        ::

          'Pi4', 'blinka', 'ft232h'

        The `'Pi4'` backend only works on the Raspberry Pis. The `'blinka'`
        backend requires installing Adafruit's modules, but is compatible with
        and maintained on a wide variety of boards. The `'ft232h'` backend
        allows controlling the GPIO from a PC using Adafruit's FT232H USB to
        I2C adapter. See :ref:`Crappy for embedded hardware` for details.
      ft232h_ser_num: If backend is `'ft232h'`, the serial number of the FT232H
        to use for communication.
    """

    # Checking that the backend is valid
    if not isinstance(backend, str) or backend not in gpio_switch_backends:
      raise ValueError("backend should be in {}".format(gpio_switch_backends))
    self._backend = backend

    # Checking that the given pin is valid for the selected backend
    if backend == 'blinka' and not hasattr(board, pin_out):
      raise TypeError(f'{pin_out} is not a valid pin using the blinka backend '
                      f'on this board !')
    elif backend == 'ft232h' and pin_out not in ft232h_pin_nr:
      raise TypeError(f'{pin_out} is not a valid pin for the ft232h backend !')
    elif backend == 'Pi4' and (not isinstance(pin_out, int) or
                               pin_out not in range(2, 28)):
      raise ValueError('pin_out should be an integer between 2 and 28 when '
                       'using the Pi4 backend !')

    # Starting the USB server (doesn't do anything if backend is not ft232h)
    Usb_server.__init__(self,
                        serial_nr=ft232h_ser_num if ft232h_ser_num else '',
                        backend=backend)
    InOut.__init__(self)
    current_file, block_number, command_file, answer_file, block_lock, \
        current_lock = super().start_server()

    # Instantiating the pin object
    if backend == 'Pi4':
      self._pin_out = pin_out
    elif backend == 'ft232h':
      self._pin_out = pin_out
      self._ft232h = ft232h(mode='GPIO_only',
                            block_number=block_number,
                            current_file=current_file,
                            command_file=command_file,
                            answer_file=answer_file,
                            block_lock=block_lock,
                            current_lock=current_lock,
                            serial_nr=ft232h_ser_num)
    elif backend == 'blinka':
      self._pin_out = digitalio.DigitalInOut(getattr(board, pin_out))

  def open(self) -> None:
    """Sets the GPIO."""

    if self._backend == 'Pi4':
      GPIO.setmode(GPIO.BCM)
      GPIO.setup(self._pin_out, GPIO.OUT)

    elif self._backend == 'blinka':
      self._pin_out.direction = digitalio.Direction.OUTPUT

  def set_cmd(self, *cmd: int) -> None:
    """Drives the GPIO according to the command.

    Args:
      cmd (:obj:`int`): 1 for driving the GPIO high, 0 for driving it low
    """

    if cmd[0] not in [0, 1]:
      raise ValueError("The GPIO input can only be 0 or 1")

    if self._backend == 'Pi4':
      GPIO.output(self._pin_out, cmd[0])
    elif self._backend == 'blinka':
      self._pin_out.value = cmd[0]
    elif self._backend == 'ft232h':
      self._ft232h.set_gpio(self._pin_out, cmd[0])

  def close(self) -> None:
    """Releases the GPIO."""

    if self._backend == 'Pi4':
      GPIO.cleanup()
    elif self._backend == 'blinka':
      self._pin_out.deinit()
    elif self._backend == 'ft232h':
      self._ft232h.close()
