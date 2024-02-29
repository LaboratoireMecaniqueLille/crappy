# coding: utf-8

from typing import Optional
import logging
from  warnings import warn

from .meta_actuator import Actuator
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


class SchneiderMDrive23(Actuator):
  """This class can drive Schneider Electric MDrive 23 stepper motor in speed
  and in position.

  It communicates with the motor over a serial connection.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *CM_drive* to *SchneiderMDrive23*
  """

  def __init__(self,
               port: str = '/dev/ttyUSB0',
               baudrate: int = 9600) -> None:
    """Sets the instance attributes and initializes the parent class.

    Args:
      port: The path to the serial port to open for the serial connection.
      baudrate: The baudrate to use for serial communication.
    """

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    self._ser = None

    super().__init__()

    self._port = port
    self._baudrate = baudrate

  def open(self) -> None:
    """Opens the serial connection to the stepper motor."""

    self.log(logging.INFO, f"Opening the serial port {self._port} with "
                           f"baudrate {self._baudrate}")
    self._ser = serial.Serial(self._port, self._baudrate, timeout=0.1)

  def set_speed(self, speed: float) -> None:
    """Sets the target speed of the stepper motor.

    Args:
      speed: The target speed to set, in `mm/min`.
    """

    # Closing and reopening to get rid of errors
    self._ser.close()
    self._ser.open()

    # Sending the command only if it's below the maximum allowed value
    if abs(speed) < 1000000:
      self.log(logging.DEBUG, f"Writing b'SL {int(speed)}\\r' to port "
                              f"{self._port}")
      self._ser.write(f'SL {int(speed)}\r')
      self._ser.read(self._ser.inWaiting())
    else:
      self.log(logging.WARNING, "Maximum speed exceeded, not setting speed")

  def set_position(self, position: float, _: Optional[float]) -> None:
    """Sets the target position for the stepper motor.

    Args:
      position: The target position to reach, in `mm`.
      _: The speed argument is ignored.

        .. versionchanged:: 2.0.0 *speed* is now a mandatory argument

    .. versionremoved:: 2.0.0 *motion_type* argument
    """

    # Closing and reopening to get rid of errors
    self._ser.close()
    self._ser.open()

    # Sending the position command
    self.log(logging.DEBUG, f"Writing b'MR {int(position)}\\r' to port "
                            f"{self._port}")
    self._ser.write(f'MR {int(position)}\r')
    self._ser.readline()

  def get_position(self) -> float:
    """Reads, displays and returns the current position in `mm`.
    
    .. versionchanged:: 1.5.2 renamed from *get_pos* to *get_position*
    """

    # Closing and reopening to get rid of errors
    self._ser.close()
    self._ser.open()

    # Asking for a position reading
    self.log(logging.DEBUG, f"Writing b'PR P \\r' to port {self._port}")
    self._ser.write('PR P \r')
    pfb = self._ser.readline()
    self.log(logging.DEBUG, f"Read {pfb} from port {self._port}")
    pfb1 = self._ser.readline()
    self.log(logging.DEBUG, f"Read {pfb1} from port {self._port}")

    return int(pfb1)

  def stop(self) -> None:
    """Sends a command for stopping the motor."""

    if self._ser is not None:
      # Closing and reopening to get rid of errors
      self._ser.close()
      self._ser.open()

      self.log(logging.DEBUG, f"Writing b'SL 0\\r' to port {self._port}")
      self._ser.write('SL 0\r')

  def close(self) -> None:
    """Close the serial connection."""

    if self._ser is not None:
      self.log(logging.INFO, f"Closing the serial port {self._port}")
      self._ser.close()
