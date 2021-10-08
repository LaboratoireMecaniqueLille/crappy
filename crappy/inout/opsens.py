# coding: utf-8

from time import time
from .inout import InOut
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


class Opsens(InOut):
  """Sensor class for opsens conditioner."""

  def __init__(self, device: str = '/dev/ttyUSB0') -> None:
    self.device = device

  def open(self) -> None:
    self.s = serial.Serial(port=self.device, baudrate=57600)
    self.send_cmd("meas:rate min")

  def close(self) -> None:
    self.s.close()

  def get_data(self) -> tuple:
    return time(), float(self.send_cmd("ch1:data? 1")[:-3])

  def read_reply(self) -> str:
    r = ''
    while r[-2:] != '\x04\n':
      r += self.s.read()
    return r

  def send_cmd(self, cmd: str) -> str:
    if '\n' in cmd:
      r = None
      for c in cmd.split('/n'):
        r = self.send_cmd(c)
      return r
    if cmd:
      self.s.write(cmd + '\n')
    return self.read_reply()
