# coding: utf-8

import logging
from ..actuator import Actuator
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


class Biaxe(Actuator):
  """This class can drive a Kollmorgen ServoStar 300 conditioner in speed.

  It is used at the LaMcube for driving a bi-axial tensile test machine, hence
  its name. The :class:`~crappy.actuator.ServoStar300` Actuator can drive the
  same hardware, but only in position.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               port: str = '/dev/ttyUSB0',
               baudrate: int = 38400,
               timeout: float = 1) -> None:
    """Sets the instance attributes.

    Args:
      port: Path to the serial port to use for communication, e.g
        `'/dev/ttyS4'`.
      baudrate: The baud rate for the serial communication.
      timeout: The timeout for the serial communication.
    """

    self._ser = None

    super().__init__()

    self._port = port
    self._baudrate = baudrate
    self._timeout = timeout

    self._speed = None

  def open(self) -> None:
    """Opens the serial connection to the ServoStar."""

    self.log(logging.INFO, f"Opening the serial port {self._port} with "
                           f"baudrate {self._baudrate}")
    self._ser = serial.Serial(self._port, self._baudrate,
                              serial.EIGHTBITS, serial.PARITY_EVEN,
                              serial.STOPBITS_ONE, self._timeout)
    self._clear_errors()

  def set_speed(self, speed: float) -> None:
    """Sets a displacement speed for the motor. A value of `1` corresponds to
    `0.002 mm/s`"""

    speed = int(speed / 0.002)

    if speed != self._speed:
      self.log(logging.DEBUG, f"Writing b'J{ {speed}}\\r\\n' on port "
                              f"{self._port}")
      self._ser.write(f'J {speed}\r\n'.encode('ASCII'))
      self._speed = speed

  def stop(self) -> None:
    """Sets the speed of the motor to `0`.
    
    .. versionadded:: 2.0.0
    """

    if self._ser is not None:
      self.set_speed(0)

  def close(self) -> None:
    """Closes the serial connection to the ServoStar."""

    if self._ser is not None:
      self.log(logging.INFO, f"Closing the serial port {self._port}")
      self._ser.close()

  def _clear_errors(self) -> None:
    """Clears the errors on the ServoStar."""

    self.log(logging.DEBUG, f"Writing b'CLRFAULT\\r\\n' on port {self._port}")
    self._ser.write(b"CLRFAULT\r\n")
    self.log(logging.DEBUG, f"Writing b'OPMODE 0\\r\\n EN\\r\\n' on port "
                            f"{self._port}")
    self._ser.write(b"OPMODE 0\r\n EN\r\n")
