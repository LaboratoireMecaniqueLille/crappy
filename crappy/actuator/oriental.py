# coding: utf-8

from typing import Optional
from .actuator import Actuator
from .._global import OptionalModule

try:
  from serial import Serial
except (ModuleNotFoundError, ImportError):
  Serial = OptionalModule("pyserial")


class Oriental(Actuator):
  """This class can drive an Oriental Motor's ARD-K stepper motor in speed or
  in position.

  It communicates with the stepper motor over a serial connection. This class
  was designed so that the :ref:`Machine` block drives several of its instances
  at a time, corresponding to different axes to drive.
  """

  def __init__(self,
               baudrate: int = 115200,
               port: str = '/dev/ttyUSB0',
               gain: float = 1/.07) -> None:
    """Sets the instance attributes and initializes the parent class.

    Args:
      baudrate: the baudrate to use for serial communication.
      port: The path to the serial port to use for communication.
      gain: The gain to apply to speed commands, in `mm/min`. The default value
        corresponds to `0.07mm/min` for a command value of `1`.
    """

    super().__init__()

    self._baudrate = baudrate
    self._port = port
    self._gain = gain

    self._prev_set_speed = 0

  def open(self) -> None:
    """Opens the serial connection to the motor and initializes the motor."""

    # Opening the serial connection to the actuator
    self._ser = Serial(self._port, baudrate=self._baudrate, timeout=0.1)

    # Checking which of the four motors is the one connected to the chosen port
    for i in range(1, 5):
      self._ser.write(f"TALK{i}\n".encode())
      ret = self._ser.readlines()

      # Printing the connected motor
      if f"{i}>".encode() in ret:
        motors = ['A', 'B', 'C', 'D']
        print(f"Motor connected to port {self._port} is {motors[i-1]}")
        break

    self._clear_errors()

    # Setting the acceleration and deceleration
    self._ser.write(b"TA .1\n")
    self._ser.write(b"TD .1\n")

  def set_speed(self, cmd: float) -> None:
    """Sets the target speed for the motor.

    Also manages the sign of the speed, i.e. the direction of the movement.
    Features a check to avoid sending the same speed command multiple times.

    Args:
      cmd: The target speed value, no units. The actual speed reached by the
        motor in `mm/min` depends on the gain that was set.
    """

    # Applying the gain and clamping the value to match the limits
    speed = min(100, int(abs(cmd * self._gain) + .5))

    # A speed of zero means a stop
    if speed == 0:
      self.stop()
      return

    # Taking the sign into consideration
    sign = int(cmd / abs(cmd))
    signed_speed = sign * speed

    # If the value is the same as the previous, do nothing
    if signed_speed == self._prev_set_speed:
      return

    # Stopping the motor if the direction changed
    dir_chg = self._prev_set_speed * sign < 0
    if dir_chg:
      self.stop()

    # Writing the target speed value
    self._ser.write(f'VR {abs(speed)}\n'.encode())
    # Writing the target direction
    if sign > 0:
      self._ser.write(b"MCP\n")
    else:
      self._ser.write(b"MCN\n")

    # Storing the written value
    self._prev_set_speed = signed_speed

  def set_position(self,
                   position: float,
                   speed: Optional[float] = None) -> None:
    """Sets the target position for the motor.

    Args:
      position: The target position to reach, in arbitrary units.
      speed: The speed to use for reching the target position, in arbitrary
        units.
    """

    if speed is None:
      raise ValueError("The Oriental actuator needs both a position and a "
                       "speed command when driven in position mode !")

    self._ser.write(f'VR {abs(speed)}'.encode())
    self._ser.write(f'MA {position}'.encode())

  def get_position(self) -> float:
    """Reads and returns the current position of the motor."""

    # Sending the read command
    self._ser.flushInput()
    self._ser.write(b'PC\n')
    self._ser.readline()

    # Reading the position and returning it
    actuator_pos = str(self._ser.readline())
    try:
      return float(actuator_pos[4:-3])
    except ValueError:
      return 0

  def stop(self) -> None:
    """Sends a command for stopping the motor."""

    self._ser.write(b"SSTOP\n")

  def close(self) -> None:
    """Closes the serial connection to the actuator."""

    self._ser.close()

  def _clear_errors(self) -> None:
    """Sends a command for clearing any serial error on the actuator."""

    self._ser.write(b"ALMCLR\n")
