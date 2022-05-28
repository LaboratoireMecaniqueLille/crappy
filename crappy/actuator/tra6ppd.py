# coding: utf-8

from time import sleep
from re import findall
from .actuator import Actuator
from .._global import OptionalModule

try:
  from serial import Serial
except (ModuleNotFoundError, ImportError):
  Serial = OptionalModule('pyserial')


class Tra6ppd(Actuator):
  """Drives the TRA6PPD linear actuator in position.

  Warning:
    This actuator cannot handle a high serial messages rate. It is recommended
    to set the frequency of the corresponding :ref:`Machine` block to a few
    dozen Hz at most.

  Note:
    This actuator ignores new position commands while it is moving.
  """

  def __init__(self,
               baudrate: int = 57600,
               port: str = '/dev/ttyUSB0') -> None:
    """Sets the instance attributes.

    Args:
      baudrate: The serial baud rate.
      port: Path to the serial port to connect to.
    """
    self._max_speed = 0.2
    self._min_position = 0
    self._max_position = 6
    self._ser = Serial(port, baudrate=baudrate, timeout=0.1)

  def open(self) -> None:
    """Resets the device and performs homing."""

    # First, reset the device
    self._ser.write(b'1RS\r\n')
    sleep(5)

    # Then, perform homing (may take up to 45s)
    self._ser.write(b'1OR\r\n')
    for i in range(5, 0, -1):
      print(f"[TRA6PPD] Performing homing, {10 * i} seconds left.")
      sleep(10)

  def close(self) -> None:
    """Just closes the serial port."""

    self._ser.close()

  def stop(self) -> None:
    """Stops the motor and sets the device to "disable" state."""

    self._ser.write(b'ST\r\n')
    self._ser.write(b'1MMO\r\n')

  def set_position(self,
                   position: float,
                   speed: float = 0.2) -> None:
    """Sends the actuator a command to reach a given position.

    The command is ignored if the actuator is already moving.

    Args:
      position: The position to reach. Should be between 0 and 6 mm.
      speed: The speed at which the actuator should move. Should be between 0
        and 0.2 mm/s.
    """

    # Clamping the speed command into the allowed interval
    speed_clamped = max(min(self._max_speed, speed), 0)

    # Sending the speed command
    speed_cmd = f'1VA{speed_clamped:.5f}\r\n'
    self._ser.write(speed_cmd.encode())

    # Ignoring the position command if it is out of range
    if not self._min_position <= position <= self._max_position:
      print(f"[TRA6PPD] WARNING : The requested position {position} is out of "
            f"range ! Ignoring.")
    else:
      # Sending the position command
      pos_command = f'1PA{position:.5f}\r\n'
      self._ser.write(pos_command.encode())

  def get_position(self) -> float:
    """Reads the current position.

    Returns:
      Current position of the motor.
    """

    # Sending the read command
    self._ser.write(b'1TP?\r\n')
    # Using a regular expression to parse the answer
    ans = self._ser.readline().decode()
    actuator_pos = float(findall(r'\d\.\d+', ans)[0])
    return actuator_pos
