# coding: utf-8

from time import time
from typing import List
from .inout import InOut
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


class Opsens(InOut):
  """This class allows reading data from an Opsens PicoSens fiber optics signal
  conditioner.

  It can read data from various fiber optics sensors like temperature,
  pressure, position or strain.
  """

  def __init__(self,
               device: str = '/dev/ttyUSB0') -> None:
    """Sets the arg and initializes the parent class.

    Args:
      device: Address of the serial connection for communicating with the
        PicoSens.
    """

    super().__init__()

    self._addr = device
    self._dev = None

  def open(self) -> None:
    """Opens the serial connection and configures the PicoSens."""

    self._dev = serial.Serial(port=self._addr, baudrate=57600)
    self._send_cmd("meas:rate min")

  def get_data(self) -> List[float]:
    """Reads data from the PicoSens and returns it."""

    return [time(), float(self._send_cmd("ch1:data? 1")[:-3])]

  def close(self) -> None:
    """Closes the serial connection if it was opened."""

    if self._dev is not None:
      self._dev.close()

  def _send_cmd(self, cmd: str) -> str:
    """Sends a command and returns the received answer."""

    self._dev.write(cmd + '\n')
    return self._dev.read_until(b'\x04\n').decode()
