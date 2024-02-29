# coding: utf-8

from time import sleep
from re import findall
from typing import Optional
import logging
from  warnings import warn

from .meta_actuator import Actuator
from .._global import OptionalModule

try:
  from serial import Serial
except (ModuleNotFoundError, ImportError):
  Serial = OptionalModule('pyserial')


class NewportTRA6PPD(Actuator):
  """Drives the Newport TRA6PPD linear actuator in position.

  Warning:
    This actuator cannot handle a high serial messages rate. It is recommended
    to set the frequency of the corresponding :class:`~crappy.blocks.Machine`
    Block to a few dozen `Hz` at most.

  Note:
    This Actuator ignores new position commands while it is moving.
  
  .. versionadded:: 1.5.10
  .. versionchanged:: 2.0.0 renamed from *TRA6PPD* to *NewportTRA6PPD*
  """

  def __init__(self,
               baudrate: int = 57600,
               port: str = '/dev/ttyUSB0') -> None:
    """Sets the instance attributes and initializes the parent class.

    Args:
      baudrate: The baudrate for the serial connection.
      port: Path to the port to use for serial communication.
    """

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    self._ser = None

    super().__init__()

    self._max_speed = 0.2
    self._min_position = 0
    self._max_position = 6

    self._baudrate = baudrate
    self._port = port

  def open(self) -> None:
    """Resets the device and performs homing."""

    self.log(logging.INFO, f"Opening the serial port {self._port} with "
                           f"baudrate {self._baudrate}")
    self._ser = Serial(self._port, baudrate=self._baudrate, timeout=0.1)

    # First, reset the device
    self.log(logging.DEBUG, f"Writing b'1RS\\r\\n' to port {self._port}")
    self._ser.write(b'1RS\r\n')
    sleep(5)

    # Then, perform homing (may take up to 45s)
    self.log(logging.DEBUG, f"Writing b'10R\\r\\n' to port {self._port}")
    self._ser.write(b'1OR\r\n')
    for i in range(5, 0, -1):
      self.log(logging.INFO, f"Performing homing, {10 * i} seconds left")
      sleep(10)

  def set_position(self, position: float, speed: Optional[float]) -> None:
    """Sends the actuator a command to reach a given position.

    The command is ignored if the actuator is already moving.

    Args:
      position: The position to reach. Should be between `0` and `6 mm`.
      speed: The speed at which the actuator should move. Should be between `0`
        and `0.2 mm/s`. If :obj:`None` is received, the default is `0.2 mm/s`.
    """

    if speed is None:
      speed = 0.2

    # Clamping the speed command into the allowed interval
    speed_clamped = max(min(self._max_speed, speed), 0)

    # Sending the speed command
    self.log(logging.DEBUG, f"Writing b'1VA{speed_clamped:.5f}\\r\\n' to port "
                            f"{self._port}")
    self._ser.write(f'1VA{speed_clamped:.5f}\r\n'.encode())

    # Ignoring the position command if it is out of range
    if not self._min_position <= position <= self._max_position:
      self.log(logging.WARNING, f"The requested position {position} is out of "
                                f"range ! Ignoring")
    else:
      # Sending the position command
      self.log(logging.DEBUG, f"Writing b'1PA{position:.5f}\\r\\n' to port "
                              f"{self._port}")
      self._ser.write(f'1PA{position:.5f}\r\n'.encode())

  def get_position(self) -> float:
    """Reads the current position.

    Returns:
      Current position of the motor.
    """

    # Sending the read command
    self.log(logging.DEBUG, f"Writing b'1TP?\\r\\n' to port {self._port}")
    self._ser.write(b'1TP?\r\n')

    # Using a regular expression to parse the answer
    ret = self._ser.readline()
    self.log(logging.DEBUG, f"Read {ret} on port {self._port}")
    return float(findall(r'\d\.\d+', ret.decode())[0])

  def close(self) -> None:
    """Closes the serial port."""

    if self._ser is not None:
      self.log(logging.INFO, f"Closing the serial port {self._port}")
      self._ser.close()

  def stop(self) -> None:
    """Stops the motor and sets the device to "disable" state."""

    if self._ser is not None:
      self.log(logging.DEBUG, f"Writing b'ST\\r\\n' to port {self._port}")
      self._ser.write(b'ST\r\n')
      self.log(logging.DEBUG, f"Writing b'1MMO\\r\\n' to port {self._port}")
      self._ser.write(b'1MMO\r\n')
