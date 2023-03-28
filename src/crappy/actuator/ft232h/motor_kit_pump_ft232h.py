# coding: utf-8

from struct import pack_into
from time import sleep
from typing import Union, List
import logging

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


class DCMotorHatFT232H:
  """Class for driving Adafruit's DC motor HAT over an FT232H.

  This class serves as a basis for building Actuators in Crappy, but is not an
  Actuator itself. It is used by the :class:`MotorKitPumpFT232H` Actuator.

  Note:
    This device can also drive stepper motors, but this feature isn't included
    here.
  """

  def __init__(self,
               motor_nrs: List[int],
               bus: FT232H,
               device_address: int = 0x60) -> None:
    """Resets the HAT and initializes it.

    Args:
      motor_nrs: The list of the motors to drive. Its elements should be
        integers between 1 and 4, corresponding to the indexes of the motors to
        drive.
      bus: The I2C commands are sent by the corresponding :class:`FT232HServer`
        instance, in the situation when the hat is controlled from a PC through
        an FT232H.
      device_address: The I2C address of the HAT. The default address is
        `0x60`, but it is possible to change this setting by cutting traces on
        the board.
    """

    if not all(i in range(1, 5) for i in motor_nrs):
      raise ValueError("The DC motor hat can only drive up to 4 DC motors at "
                       "a time !")

    if not isinstance(device_address, int):
      raise TypeError("device_address should be an integer between 0 and 127.")
    self._address = device_address

    self._bus = bus
    self._buf = bytearray(4)

    # Reset
    self._write_i2c(0x00, [0x00])
    sleep(0.01)

    # Sleep & restart, also setting the frequency to 1526Hz
    self._write_i2c(0x00, [0x10])
    prescale = int(25E6 / 4096.0 / 1600.0 + 0.5)
    self._write_i2c(0xFE, [prescale])
    sleep(0.01)
    self._write_i2c(0x00, [0x00])
    sleep(0.01)
    self._write_i2c(0x00, [0xA0])

    # Initializing the motors
    for i in motor_nrs:
      self._write_i2c(motor_hat_ctrl[i], motor_hat_0xFF)

  def set_motor(self, nr: int, cmd: float) -> None:
    """Sets the PWMs associated with a given motor.

    Args:
      nr: The index of the motor to drive.
      cmd: The command as a :obj:`float`, that should be between -1 and 1.
    """

    # Validity checks
    if nr not in range(1, 5):
      raise ValueError("It is only possible to drive up to 4 motors.")
    if not -1.0 <= cmd <= 1.0:
      raise ValueError("The command for the motor should be between -1 and 1.")

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

  def close(self) -> None:
    """Closes the I2C bus."""

    self._bus.close()

  def _write_i2c(self, register: int, buf: Union[list, bytearray]) -> None:
    """Thin wrapper to reduce verbosity."""

    self._bus.write_i2c_block_data(self._address, register, list(buf))


class MotorKitPumpFT232H(Actuator):
  """Class for controlling two DC air pumps and a valve over an FT232H.

  It uses Adafruit's DC motor HAT, defined in :class:`DCMotorHatFT232H`. The
  motor 1 controls the inflation pump, the motor 2 controls a valve, and the
  motor 3 controls a deflation pump.

  This Actuator is similar to the :ref:`Motor kit pump` one as they drive the
  same hardware, except this one sends the I2C messages through an FT232H
  device.
  """

  ft232h = True

  def __init__(self,
               device_address: int = 0x60,
               _ft232h_args: USBArgsType = tuple()) -> None:
    """Checks the validity of the arguments.

    Args:
      device_address: The I2C address of the HAT. The default address is
        `0x60`, but it is possible to change this setting by cutting traces on
        the board.
      _ft232h_args: This argument is for internal use only and should not be
        accessed by users.
    """

    self._hat = None

    super().__init__()

    (block_index, block_lock, command_file, answer_file, shared_lock,
     current_block) = _ft232h_args

    self._bus = FT232H(mode='I2C',
                       block_index=block_index,
                       current_block=current_block,
                       command_file=command_file,
                       answer_file=answer_file,
                       block_lock=block_lock,
                       shared_lock=shared_lock)

    if not isinstance(device_address, int):
      raise TypeError("device_address should be an integer between 0 and 127.")
    self._address = device_address

  def open(self) -> None:
    """Initializes the generic HAT object."""

    self.log(logging.INFO, "Opening the Motorkit with an USB connection "
                           "over FT232H")
    self._hat = DCMotorHatFT232H([1, 2, 3], self._bus, self._address)

  def set_speed(self, volt: float) -> None:
    """Inflates or deflates the setup according to the command.

    Args:
      volt: The voltage to supply to the pumps. If positive, will inflate, if
        negative will deflate, and if `0` won't do anything. The voltage is
        clamped between `-12` and `12` Volts, as it is the limit of the HAT.
    """

    volt_clamped = min(abs(volt) / motor_hat_max_volt, 1.0)

    # Stops all the motors
    if volt == 0:
      self.log(logging.DEBUG, "Setting motor 1 to 0.0")
      self._hat.set_motor(1, 0.0)
      self.log(logging.DEBUG, "Setting motor 2 to 0.0")
      self._hat.set_motor(2, 0.0)
      self.log(logging.DEBUG, "Setting motor 3 to 0.0")
      self._hat.set_motor(3, 0.0)

    # Drives the inflating pump
    elif volt > 0:
      self.log(logging.DEBUG, f"Setting motor 1 to {volt_clamped}")
      self._hat.set_motor(1, volt_clamped)
      self.log(logging.DEBUG, "Setting motor 2 to 0.0")
      self._hat.set_motor(2, 0.0)
      self.log(logging.DEBUG, "Setting motor 3 to 0.0")
      self._hat.set_motor(3, 0.0)

    # Drives the deflating pump and opens the valve
    elif volt < 0:
      self.log(logging.DEBUG, "Setting motor 1 to 0.0")
      self._hat.set_motor(1, 0.0)
      self.log(logging.DEBUG, "Setting motor 2 to 1.0")
      self._hat.set_motor(2, 1.0)
      self.log(logging.DEBUG, f"Setting motor 3 to {volt_clamped}")
      self._hat.set_motor(3, volt_clamped)

  def stop(self) -> None:
    """Simply sets the command to `0` to stop the pump."""

    if self._hat is not None:
      self.set_speed(0)

  def close(self) -> None:
    """Stops the pumps and closes the HAT object."""

    if self._hat is not None:
      self.log(logging.INFO, "Closing the connection to the Motorkit")
      self._hat.close()
