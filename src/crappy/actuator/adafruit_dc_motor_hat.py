# coding: utf-8

from struct import pack_into
from time import sleep
from typing import Union, Literal
import logging
from  warnings import warn

from .meta_actuator import Actuator
from .._global import OptionalModule

try:
  from adafruit_motorkit import MotorKit
except (ImportError, ModuleNotFoundError):
  MotorKit = OptionalModule('adafruit_motorkit',
                            'Adafrfuit Motorkit module (adafruit_motorkit) is '
                            'required to use this actuator')

try:
  import board
except (ImportError, ModuleNotFoundError):
  board = OptionalModule('board', 'Blinka is necessary to use the I2C bus')

try:
  from smbus2 import SMBus
except (ImportError, ModuleNotFoundError):
  SMBus = OptionalModule('smbus2')

motor_hat_ctrl = {1: 0x26,
                  2: 0x3A,
                  3: 0x0E,
                  4: 0x22}

motor_hat_pos = {1: 0x2A,
                 2: 0x32,
                 3: 0x12,
                 4: 0x1A}

motor_hat_neg = {1: 0x2E,
                 2: 0x36,
                 3: 0x16,
                 4: 0x1E}

motor_hat_0xFF = [0x00, 0x10, 0x00, 0x00]
motor_hat_backends = ['Pi4', 'blinka']
motor_hat_max_volt = 12


class DCMotorHat(Actuator):
  """Class for driving Adafruit's DC motor HAT.

  It can drive up to four DC motors in speed only. The acquisition of the speed
  has not been implemented so far. It can either rely on Adafruit's Blinka
  library, or on :mod:`smbus2` if used from a Raspberry Pi.

  Important:
    As this Actuator can drive up to 4 motors simultaneously, it takes a
    :obj:`tuple` as a command, see :meth:`set_speed`. Regular Actuators receive
    their commands as :obj:`float`. A :class:`~crappy.modifier.Modifier` can be
    used for converting a :obj:`float` command from a
    :class:`~crappy.blocks.Generator` to a :obj:`tuple`.

  Note:
    The DC Motor Hat can also drive stepper motors, but this feature isn't
    included here.
  
  .. versionadded:: 2.0.0
  """

  def __init__(self,
               backend: Literal['Pi4', 'blinka'],
               device_address: int = 0x60,
               i2c_port: int = 1) -> None:
    """Checks the validity of the arguments.

    Args:
      backend: Should be one of :
        ::

          'Pi4', 'blinka'

        The `'Pi4'` backend is optimized but only works on boards supporting
        the :mod:`smbus2` module, like the Raspberry Pis. The `'blinka'`
        backend may be less performant and requires installing
        :mod:`adafruit-circuitpython-motorkit` and :mod:`Adafruit-Blinka`, but
        these modules are compatible with and maintained on a wide variety of
        boards.
      device_address: The I2C address of the HAT. The default address is
        `0x60`, but it is possible to change this setting by cutting traces on
        the board.
      i2c_port: The I2C port over which the HAT should communicate. On most
        Raspberry Pi models the default I2C port is `1`.
    """

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    self._bus = None
    self._buf = bytearray(4)

    if not isinstance(backend, str) or backend not in motor_hat_backends:
      raise ValueError("backend should be in {}".format(motor_hat_backends))
    self._backend = backend

    super().__init__()

    if not isinstance(device_address, int):
      raise TypeError("device_address should be an integer between 0 and 127.")
    self._address = device_address

    if not isinstance(i2c_port, int):
      raise TypeError("i2c_port should be an integer !")
    self._port = i2c_port

  def open(self) -> None:
    """Opens the connection to the motor hat and initializes it."""

    if self._backend == 'blinka':
      self.log(logging.INFO, "Opening with the Adafruit library")
      self._bus = MotorKit(i2c=board.I2C())

    elif self._backend == 'Pi4':
      self.log(logging.INFO, "Opening with the SMBus library")

      self._bus = SMBus(self._port)

      # Reset
      self._write_i2c(0x00, [0x00])
      sleep(0.01)

      # Sleep & restart, also setting the frequency to 1526Hz
      self._write_i2c(0x00, [0x10])
      self._write_i2c(0xFE, [int(25E6 / 4096.0 / 1600.0 + 0.5)])
      sleep(0.01)
      self._write_i2c(0x00, [0x00])
      sleep(0.01)
      self._write_i2c(0x00, [0xA0])

      # Initializing the motors
      for i in range(1, 5):
        self._write_i2c(motor_hat_ctrl[i], motor_hat_0xFF)

  def set_speed(self, cmd: tuple[int, float]) -> None:
    """Sets the desired voltage on the selected motor.

    The provided voltage should be between `-12` and `12V` which are the limits
    of the motor hat. If not, it will be silently clamped in this interval.

    Warning:
      Unlike most Actuators, this one takes :obj:`tuple` as commands instead
      of :obj:`float`.

    Args:
      cmd: A :obj:`tuple` containing first the index of the motor to drive as
        an :obj:`int` between 1 and 5, and second the voltage to apply to the
        selected motor as a :obj:`float`.
    """

    try:
      motor_nr, volts = cmd
    except (TypeError, ValueError):
      raise ValueError("The DCMotorHat receives commands as tuples, with first"
                       " the index of the motor and second the voltage "
                       "command")

    if motor_nr not in range(1, 5):
      raise ValueError("The index of the motor to drive should be an integer "
                       "between 1 and 5")

    volt_clamped = min(abs(volts) / motor_hat_max_volt, 1.0)
    self.log(logging.DEBUG, f"Setting motor {motor_nr} to {volt_clamped}V")
    self._set_motor(motor_nr, volt_clamped)

  def stop(self) -> None:
    """Simply sets the command to `0` to stop the motors."""

    if self._bus is not None:
      for i in range(1, 5):
        self.set_speed((i, 0))

  def close(self) -> None:
    """Closes the I2C connection to the motor hat."""

    if self._bus is not None and self._backend == 'Pi4':
      self.log(logging.INFO, "Closing the connection to the Motor Hat")
      self._bus.close()

  def _set_motor(self, nr: int, cmd: float) -> None:
    """Sets the PWMs associated with a given motor.

    Args:
      nr: The index of the motor to drive.
      cmd: The command as a :obj:`float`, that should be between -1 and 1.
    """

    if self._backend == 'Pi4':

      # Special settings for the extreme values
      if cmd == 0:
        self._write_i2c(motor_hat_pos[nr], motor_hat_0xFF)
        self._write_i2c(motor_hat_neg[nr], motor_hat_0xFF)
      elif cmd == 1.0:
        self._write_i2c(motor_hat_pos[nr], motor_hat_0xFF)
        self._write_i2c(motor_hat_neg[nr], [0x00 for _ in range(4)])
      elif cmd == -1.0:
        self._write_i2c(motor_hat_pos[nr], [0x00 for _ in range(4)])
        self._write_i2c(motor_hat_neg[nr], motor_hat_0xFF)

      # The positive line is driven for positive commands, and reversely
      elif cmd > 0:
        duty_cycle = (int(0xFFFF * cmd) + 1) >> 4
        pack_into("<HH", self._buf, 0, *(0, duty_cycle))
        self._write_i2c(motor_hat_pos[nr], self._buf)
        self._write_i2c(motor_hat_neg[nr], [0x00 for _ in range(4)])
      else:
        duty_cycle = (int(0xFFFF * abs(cmd)) + 1) >> 4
        pack_into("<HH", self._buf, 0, *(0, duty_cycle))
        self._write_i2c(motor_hat_pos[nr], [0x00 for _ in range(4)])
        self._write_i2c(motor_hat_neg[nr], self._buf)

    # Blinka handles all the job internally
    elif self._backend == 'blinka':
      setattr(getattr(self._bus, f'motor{nr}'), 'throttle', cmd)

  def _write_i2c(self, register: int, buf: Union[list, bytearray]) -> None:
    """Thin wrapper to reduce verbosity."""

    self._bus.write_i2c_block_data(self._address, register, list(buf))
