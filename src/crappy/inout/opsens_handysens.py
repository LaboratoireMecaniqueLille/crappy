# coding: utf-8

from time import time
import logging
from  warnings import warn

from .meta_inout import InOut
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


class HandySens(InOut):
  """This class allows reading data from an OpSens HandySens fiber optics
  signal conditioner.

  It can read data from various fiber optics sensors like temperature,
  pressure, position or strain.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Opsens* to *HandySens*
  """

  def __init__(self,
               device: str = '/dev/ttyUSB0') -> None:
    """Sets the argument and initializes the parent class.

    Args:
      device: Address of the serial connection for communicating with the
        OpSens.
    """

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    self._dev = None

    super().__init__()

    self._addr = device

  def open(self) -> None:
    """Opens the serial connection and configures the OpSens."""

    self.log(logging.INFO, f"Opening the serial connection on port "
                           f"{self._addr} with baudrate 57600")
    self._dev = serial.Serial(port=self._addr, baudrate=57600, timeout=0.1)
    self._send_cmd("meas:rate min")

  def get_data(self) -> list[float]:
    """Reads data from the OpSens and returns it."""

    return [time(), float(self._send_cmd("ch1:data? 1")[:-3])]

  def close(self) -> None:
    """Closes the serial connection, if it was opened."""

    if self._dev is not None:
      self.log(logging.INFO, f"Closing the serial connection on port "
                             f"{self._addr}")
      self._dev.close()

  def _send_cmd(self, cmd: str) -> str:
    """Wrapper for sending a command and returning the received answer."""

    self.log(logging.DEBUG, f"Writing b'{cmd}\\n' to port {self._addr}")
    self._dev.write(cmd + '\n')
    ret = self._dev.read_until(b'\x04\n').decode()
    self.log(logging.DEBUG, f"Read {ret} on port {self._addr}")
    return ret
