# coding: utf-8

from struct import pack_into
from time import sleep
from typing import Union, Optional
import logging
from warnings import warn

from ..meta_actuator import Actuator
from ...tool.ft232h import FT232HServer as FT232H, USBArgsType

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
motor_hat_max_volt = 12


class DCMotorHatFT232H(Actuator):
  """Class for driving Adafruit's DC motor HAT via an FT232H USB to I2C
  converter.

  It can drive up to four DC motors in speed only. The acquisition of the speed
  has not been implemented so far. It implements the same functionality as the
  :class:`~crappy.actuator.DCMotorHat`, but communicates with the hat over USB
  via an FT232H device.

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

  ft232h = True

  def __init__(self,
               device_address: int = 0x60,
               _ft232h_args: USBArgsType = tuple()) -> None:
    """Checks the validity of the arguments and opens the connection to the
    FT232H.

    Args:
      device_address: The I2C address of the HAT. The default address is
        `0x60`, but it is possible to change this setting by cutting traces on
        the board.
      _ft232h_args: This argument is meant for internal use only and should not
        be provided by the user. It contains the information necessary for
        setting up the FT232H.
    """

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    self._bus: Optional[FT232H] = None
    self._buf: bytearray = bytearray(4)

    super().__init__()

    if not isinstance(device_address, int):
      raise TypeError("device_address should be an integer between 0 and 127.")
    self._address = device_address

    (block_index, block_lock, command_file, answer_file, shared_lock,
     current_block) = _ft232h_args

    self._bus = FT232H(mode='I2C',
                       block_index=block_index,
                       current_block=current_block,
                       command_file=command_file,
                       answer_file=answer_file,
                       block_lock=block_lock,
                       shared_lock=shared_lock)

  def open(self) -> None:
    """Opens the connection to the motor hat and initializes it."""

    self.log(logging.INFO, "Opening with a USB connection over FT232H")

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
        selected motor as a :obj:`float`."""

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

    if self._bus is not None:
      self.log(logging.INFO, "Closing the connection to the Motor Hat")
      self._bus.close()

  def _set_motor(self, nr: int, cmd: float) -> None:
    """Sets the PWMs associated with a given motor.

    Args:
      nr: The index of the motor to drive.
      cmd: The command as a :obj:`float`, that should be between -1 and 1.
    """

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

  def _write_i2c(self, register: int, buf: Union[list, bytearray]) -> None:
    """Thin wrapper to reduce verbosity."""

    self._bus.write_i2c_block_data(self._address, register, list(buf))
