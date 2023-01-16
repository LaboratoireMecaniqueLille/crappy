# coding: utf-8

from time import time
from typing import List
import logging

from .meta_inout import InOut
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


class Agilent34420a(InOut):
  """Sensor class for Agilent34420A devices.

  This class contains method to measure values of resistance or voltage on
  Agilent34420A devices.

  Note:
    May work for other devices too, but not tested.

    If you have issues with this class returning a lot of `'bad serial'`, make
    sure you have the last version of :mod:`serial`.
  """

  def __init__(self,
               mode: bytes = b"VOLT",
               device: str = '/dev/ttyUSB0',
               baudrate: int = 9600,
               timeout: float = 1) -> None:
    """Sets the args and initializes parent class.

    Args:
      mode: Desired value to measure. Should be either `b'VOLT'` or `b'RES'`.
      device: Path to the device to open, as a :obj:`str`.
      baudrate: Desired baudrate for serial communication.
      timeout: Timeout for the serial connection, as a :obj:`float`.
    """

    super().__init__()

    self._device = device
    self._baudrate = baudrate
    self._timeout = timeout
    self._mode = mode

    self._ser = None

  def open(self) -> None:
    """Opens the serial connection, resets the Agilent and configures it to the
    desired mode."""

    self.log(logging.INFO, f"Opening the serial port {self._device} with "
                           f"baudrate {self._baudrate}")
    self._ser = serial.Serial(port=self._device, baudrate=self._baudrate,
                              timeout=self._timeout)
    self.log(logging.DEBUG, f"Writing b'*RST;*CLS;*OPC?\\n' to port "
                            f"{self._device}")
    self._ser.write(b"*RST;*CLS;*OPC?\n")
    self.log(logging.DEBUG, f"Writing b'SENS:FUNC \"{self._mode}\";  \\n' "
                            f"to port {self._device}")
    self._ser.write(b"SENS:FUNC \"" + self._mode + b"\";  \n")
    self.log(logging.DEBUG, f"Writing b'SENS:{self._mode}:NPLC 2  \\n' "
                            f"to port {self._device}")
    self._ser.write(b"SENS:" + self._mode + b":NPLC 2  \n")
    self.log(logging.DEBUG, f"Writing b'SYST:REM\\n' to port {self._device}")
    self._ser.write(b"SYST:REM\n")

  def get_data(self) -> List[float]:
    """Asks the Agilent to acquire a reading and returns it, except if an error
    occurs in which case `0` is returned."""

    self.log(logging.DEBUG, f"Writing b'READ?  \\n' to port {self._device}")
    self._ser.write(b"READ?  \n")
    t = time()
    try:
      return [t, float(self._ser.readline())]
    except (serial.SerialException, ValueError):
      self._ser.flush()
      return [t, 0]

  def close(self) -> None:
    """Closes the serial port."""

    if self._ser is not None:
      self.log(logging.INFO, f"Closing the serial port {self._device}")
      self._ser.close()
