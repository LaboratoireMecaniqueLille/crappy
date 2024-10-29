# coding: utf-8

from time import time
from typing import Literal
import logging
from  warnings import warn

from .meta_inout import InOut
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


class Agilent34420a(InOut):
  """This class can read resistance and voltage values from an Agilent 34420A
  multimeter.

  It communicates over serial.

  May also work on similar devices from the same manufacturer, although that
  was not tested.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               mode: Literal[b'VOLT', b'RES'] = b"VOLT",
               device: str = '/dev/ttyUSB0',
               baudrate: int = 9600,
               timeout: float = 1) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      mode: Measurement mode, as :obj:`bytes`. Should be either `b'VOLT'` or
        `b'RES'`.
      device: Path to the serial port to open, as a :obj:`str`.
      baudrate: Desired baudrate for serial communication.
      timeout: Timeout for the serial connection, as a :obj:`float`.
    """

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    self._ser = None

    super().__init__()

    self._device = device
    self._baudrate = baudrate
    self._timeout = timeout
    self._mode = mode

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

  def get_data(self) -> list[float]:
    """Asks the Agilent to acquire a value and returns it, except if an error
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
