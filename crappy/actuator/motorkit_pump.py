# coding: utf-8

from .actuator import Actuator
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


class Motorkit_pump(Actuator):
  """The Motorkit_pump actuator is meant for controlling two DC pumps and a
  valve using Adafruit's DC hat for Raspberry Pi.

  It is a basic actuator that can only inflate, deflate or do nothing. The
  communication with the hat is over I2C.

  Note:
    May work on other boards than Raspberry Pi supporting I2C, assuming a proper
    wiring.
  """

  def __init__(self, initial_speed=0, initial_pos=0) -> None:
    """Nothing to do here."""

    super().__init__()
    self.initial_speed = initial_speed
    self.initial_pos = initial_pos
    self.kit = None

  def open(self) -> None:
    """Simply creates the Motorkit object."""

    self.rpm = self.initial_speed  # checking speed is not available, however
    # the air pump outflows 2.5LPM with 4.5V
    self.pos = self.initial_pos
    self.u = 0  # Volts
    self.kit = MotorKit(i2c=board.I2C())

  def stop(self) -> None:
    """Sets the speed to `0`."""

    self.set_speed(0)

  def close(self) -> None:
    """Just stops the motor."""

    self.stop()
    self.kit = None

  def set_speed(self, u) -> None:
    """Controls the pumps and the valve so that they either inflate or deflate.

    The input range is from `-1` (deflate at full speed) to `1` (inflate at full
    speed).

    Args:
      u: The desired inflating rate.
    """

    u_max = 12  # max allowed voltage of the pump
    if u == 0:  # shutdown
      self.kit.motor1.throttle = 0
      self.kit.motor2.throttle = 0
      self.kit.motor3.throttle = 0
    elif u > 0:  # inflate
      p = round(u / u_max, 2)  # motor only accepts values from 0 to 1
      # 1 being 12V
      if p > 1:  # command saturation
        p = 1
      self.kit.motor1.throttle = p  # inflate pump on
      self.kit.motor2.throttle = 0
      self.kit.motor3.throttle = 0
    elif u < 0:  # deflate
      u = -u
      p = round(u / u_max, 2)  # motor only accepts values from 0 to 1
      # 1 being 12V
      if p > 1:  # command saturation
        p = 1
      self.kit.motor1.throttle = 0
      self.kit.motor2.throttle = p  # deflate pump on
      self.kit.motor3.throttle = 1.0  # open valve
