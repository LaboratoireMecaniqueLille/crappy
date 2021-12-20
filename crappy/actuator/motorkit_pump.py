# coding: utf-8

from struct import pack_into
from time import sleep
from typing import Union
from .actuator import Actuator
from .._global import OptionalModule
from ..tool import ft232h_server as ft232h, Usb_server

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
motor_hat_backends = ['Pi4', 'blinka', 'ft232h']
motor_hat_max_volt = 12


class DC_motor_hat:
  """Class for driving Adafruit's DC motor HAT.

  This class serves as a basis for building Actuators in Crappy, but is not one
  itself. It is used by the :class:`Motorkit_pump` Actuator.

  Note:
    This device can also drive stepper motors, but this feature isn't included
    here.

  It is intended for Raspberry Pis but can also be used from any other device
  interfacing over I2C assuming a proper wiring.
  """

  def __init__(self,
               motor_nrs: list,
               device_address: int = 0x60,
               i2c_port: int = 1,
               bus: ft232h = None) -> None:
    """Resets the HAT and initializes it.

    Args:
      motor_nrs (:obj:`list`): The list of the motors to drive. Its elements
        should be integers between 1 and 4, corresponding to the number of the
        motors.
      device_address (:obj:`int`, optional): The I2C address of the HAT.
        The default address is `0x60`, but it is possible to change this
        setting by cutting traces on the board.
      i2c_port (:obj:`int`, optional): The I2C port over which the ADS1115
        should communicate. On most Raspberry Pi models the default I2C port is
        `1`.
      bus (:obj:`ft232h_server`, optional): If given, the I2C commands are sent
        by the corresponding :class:`ft232h_server` instance, in the situation
        when the hat is controlled from a PC through an FT232H rather than by a
        board.
    """

    if not all(i in range(1, 5) for i in motor_nrs):
      raise ValueError("The DC motor hat can only drive up to 4 DC motors at "
                       "a time !")

    if not isinstance(device_address, int):
      raise TypeError("device_address should be an integer between 0 and 127.")
    self._address = device_address

    if not isinstance(i2c_port, int):
      raise TypeError("i2c_port should be an integer !")

    if not bus:
      self._bus = SMBus(i2c_port)
    else:
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
      nr (:obj:`int`): The number of the motor to drive.
      cmd (:obj:`float`): The command, that should be between -1 and 1.
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

  def _write_i2c(self, register: int, buf: Union[list, bytearray]):
    """Thin wrapper to reduce verbosity."""

    self._bus.write_i2c_block_data(self._address, register, list(buf))


class Motorkit_pump(Usb_server, Actuator):
  """Class for controlling two DC air pumps and a valve.

  It uses Adafruit's DC motor HAT. The motor 1 controls the inflation pump,
  the motor 2 controls a valve, and the motor 3 controls a deflation pump.
  """

  def __init__(self,
               backend: str,
               device_address: int = 0x60,
               i2c_port: int = 1,
               ft232h_ser_num: str = None) -> None:
    """Checks the validity of the arguments.

    Args:
      backend (:obj:`str`): Should be one of :
        ::

          'Pi4', 'blinka', 'ft232h'

        The `'Pi4'` backend is optimized but only works on boards supporting
        the :mod:`smbus2` module, like the Raspberry Pis. The `'blinka'`
        backend may be less performant and requires installing Adafruit's
        modules, but these modules are compatible with and maintained on a wide
        variety of boards. The `'ft232h'` backend allows controlling the hat
        from a PC using Adafruit's FT232H USB to I2C adapter. See
        :ref:`Crappy for embedded hardware` for details.
      device_address (:obj:`int`, optional): The I2C address of the HAT.
        The default address is `0x60`, but it is possible to change this
        setting by cutting traces on the board.
      i2c_port (:obj:`int`, optional): The I2C port over which the HAT should
        communicate. On most Raspberry Pi models the default I2C port is
        `1`.
      ft232h_ser_num (:obj:`str`, optional): If backend is `'ft232h'`, the
        serial number of the ft232h to use for communication.
    """

    if not isinstance(backend, str) or backend not in motor_hat_backends:
      raise ValueError("backend should be in {}".format(motor_hat_backends))
    self._backend = backend

    Usb_server.__init__(self,
                        serial_nr=ft232h_ser_num if ft232h_ser_num else '',
                        backend=backend)
    Actuator.__init__(self)
    queue, block_number, namespace, command_event, \
        answer_event, next_event, done_event = super().start_server()

    if self._backend == 'ft232h':
      self._bus = ft232h(mode='I2C',
                         queue=queue,
                         namespace=namespace,
                         command_event=command_event,
                         answer_event=answer_event,
                         block_number=block_number,
                         next_block=next_event,
                         done_event=done_event,
                         serial_nr=ft232h_ser_num)
    else:
      self._bus = None

    if not isinstance(device_address, int):
      raise TypeError("device_address should be an integer between 0 and 127.")
    self._address = device_address

    if not isinstance(i2c_port, int):
      raise TypeError("i2c_port should be an integer !")
    self._port = i2c_port

  def open(self) -> None:
    """Initializes the generic HAT object."""

    if self._backend == 'blinka':
      self._hat = MotorKit(i2c=board.I2C())

      def set_motor(nr: int, cmd: float) -> None:
        setattr(getattr(self._hat, 'motor{}'.format(nr)), 'throttle', cmd)

      self._hat.set_motor = set_motor

    else:
      self._hat = DC_motor_hat([1, 2, 3], self._address, self._port, self._bus)

  def set_speed(self, volt: float) -> None:
    """Inflates or deflates the setup according to the command.

    Args:
      volt (:obj:`float`): The voltage to supply to the pumps. If positive,
        will inflate, if negative will deflate, and if 0 won't do anything. The
        voltage is clamped between -12 and 12 Volts, as it is the limit of the
        HAT.
    """

    volt_clamped = min(abs(volt) / motor_hat_max_volt, 1.0)

    # Stops all the motors
    if volt == 0:
      self._hat.set_motor(1, 0.0)
      self._hat.set_motor(2, 0.0)
      self._hat.set_motor(3, 0.0)

    # Drives the inflating pump
    elif volt > 0:
      self._hat.set_motor(1, volt_clamped)
      self._hat.set_motor(2, 0.0)
      self._hat.set_motor(3, 0.0)

    # Drives the deflating pump and opens the valve
    elif volt < 0:
      self._hat.set_motor(1, 0.0)
      self._hat.set_motor(2, 1.0)
      self._hat.set_motor(3, volt_clamped)

  def stop(self) -> None:
    """Simply sets the voltage to 0."""

    self.set_speed(0.0)

  def close(self) -> None:
    """Stops the pumps and closes the HAT object."""

    self.stop()
    if self._backend != 'blinka':
      self._hat.close()
