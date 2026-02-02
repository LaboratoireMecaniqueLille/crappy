# coding: utf-8

import logging
import time
from  warnings import warn

from .meta_inout import InOut
from .._global import OptionalModule

try:
  from pymodbus.client.serial import ModbusSerialClient
except (ModuleNotFoundError, ImportError):
  ModbusSerialClient = OptionalModule("pymodbus")

REGISTERS: dict[str, tuple[int, int]] = {"process_value": (1, 1),
                                         "setpoint": (2, 1)}


class EurothermEPC3008(InOut):
  """Driver for Eurotherm EPC3008 over Modbus RTU (serial).

  .. versionadded:: 2.0.9
  """

  def __init__(self,
               port: str,
               address: int = 1,
               timeout: float = 1.0) -> None:
    """Sets the arguments and initializes parent class.

    Args:
      port: Serial port on which to communicate with the device (e.g. "COM7" on
        Windows, "/dev/ttyUSB0" on Linux).
      address: Modbus slave address (unit id).
      timeout: Read/write timeout (seconds).
    """

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    self._client: ModbusSerialClient | None = None

    super().__init__()

    self._port = port
    self._address = address
    self._timeout = timeout

  def open(self) -> None:
    """Open the serial Modbus RTU connection."""

    self._client = ModbusSerialClient(port=self._port, baudrate=9600,
                                      bytesize=8, parity='N', stopbits=1,
                                      timeout=self._timeout, method="rtu")
    self.log(logging.INFO, f"Eurotherm EPC3008 connected on {self._port}")

  def get_data(self) -> tuple[float, float, float] | None:
    """Read process value (PV) and setpoint (SP) in degree Celsius."""

    now = time.time()

    pv_addr, pv_len = REGISTERS["process_value"]
    sp_addr, sp_len = REGISTERS["setpoint"]

    pv_resp = self._client.read_holding_registers(
        address=pv_addr, count=pv_len, slave=self._address)
    sp_resp = self._client.read_holding_registers(
        address=sp_addr, count=sp_len, slave=self._address)

    if (pv_resp.isError()
        or sp_resp.isError()
        or not hasattr(pv_resp, "registers")
        or not hasattr(sp_resp, "registers")):
      self.log(logging.WARNING, "Eurotherm read error")
      return None

    pv = float(pv_resp.registers[0])
    sp = float(sp_resp.registers[0])

    return now, pv, sp

  def set_cmd(self, cmd: float | int) -> None:
    """Write the setpoint (degree Celsius)."""

    reg_addr, _ = REGISTERS["setpoint"]
    value = int(cmd) & 0xFFFF

    resp = self._client.write_register(address=reg_addr,
                                       value=value, slave=self._address)

    if resp.isError():
      raise IOError(f"Eurotherm EPC3008 write error: {resp}")

  def close(self) -> None:
    """Close the serial Modbus RTU connection."""

    if self._client is None:
      return

    self._client.close()
    self.log(logging.INFO, f"Eurotherm EPC3008 disconnected of {self._port}")
