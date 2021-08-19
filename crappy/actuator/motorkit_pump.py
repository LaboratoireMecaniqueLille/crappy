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

  def __init__(self) -> None:
    """Not much to do here."""

    super().__init__()
    self._kit = None

  def open(self) -> None:
    """Simply creates the Motorkit object."""

    self._kit = MotorKit(i2c=board.I2C())

  def stop(self) -> None:
    """Sets the speed to `0`."""

    self.set_speed(0)

  def close(self) -> None:
    """Just stops the motor."""

    self.stop()
    self._kit = None

  def set_speed(self, volt: float) -> None:
    """Controls the pumps and the valve so that they either inflate or deflate.

    The input range is from `-12` (deflate at full speed) to `12` (inflate at
    full speed).

    Args:
      volt (:obj:`float`): The voltage to supply to the motor.
    """

    volt_max = 12  # max allowed voltage of the pump
    volt_clamped = min(abs(volt) / volt_max, 1)
    # motor only accepts values from 0 to 1, 1 being 12V

    if volt == 0:  # shutdown
      self._kit.motor1.throttle = 0
      self._kit.motor2.throttle = 0
      self._kit.motor3.throttle = 0

    elif volt > 0:  # inflate
      self._kit.motor1.throttle = volt_clamped  # inflate pump on
      self._kit.motor2.throttle = 0
      self._kit.motor3.throttle = 0

    elif volt < 0:  # deflate
      self._kit.motor1.throttle = 0
      self._kit.motor2.throttle = volt_clamped  # deflate pump on
      self._kit.motor3.throttle = 1.0  # open valve
