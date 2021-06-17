# coding: utf-8

import time
from .inout import InOut
from .._global import OptionalModule

try:
  from pijuice import PiJuice
except (ModuleNotFoundError, ImportError):
  PiJuice = OptionalModule("pijuice")


class Pijuice(InOut):
  """Block for knowing the status (plugged or unplugged) and the actual
  level of charge of the piJuice.

  Warning:
    Only available on Raspberry Pi !
  """

  def __init__(self, i2c_port: int = 1, address: int = 0x14) -> None:
    """Checks arguments validity.

    Args:
      i2c_port(:obj:`int`,optional): The I2C port over which the PiJuice should
        communicate.
      address(:obj:`int`,optional): The I2C address of the MCP9600. The default
        address is 0x14.
    """

    super().__init__()
    if not isinstance(i2c_port, int):
      raise TypeError("i2c_port should be an int")
    else:
      self.i2c_port = i2c_port
    if not isinstance(address, int):
      raise TypeError("address should be an int")
    else:
      self.address = address

  def open(self):
    self.pijuice = PiJuice(self.i2c_port, self.address)

  def get_data(self) -> list:
    """Reads the status and the charge level.

    The output is 0 if unplugged and 1 if plugged for status and the output is
    between 0 and 100 for charge.

    Returns:
      :obj:`list`: A list containing the timeframe and the output value for
      status and charge.
    """

    # Reads the battery status
    value = self.pijuice.status.GetStatus()
    # Reads the battery charge level
    charge = self.pijuice.status.GetChargeLevel()
    out = [time.time()]  # Date of data recovery
    out.append(value["data"]["powerInput5vIo"] == "PRESENT")
    out.append(charge["data"])

    return out

  def close(self):
    pass
