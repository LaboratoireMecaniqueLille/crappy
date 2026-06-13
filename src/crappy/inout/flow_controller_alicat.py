# coding: utf-8

import logging
import time
from  warnings import warn

from .meta_inout import InOut
from .._global import OptionalModule

try:
  from pymodbus.client.serial import ModbusSerialClient
  from pymodbus.constants import Endian
  from pymodbus.payload import BinaryPayloadBuilder, BinaryPayloadDecoder
except (ModuleNotFoundError, ImportError):
  ModbusSerialClient = OptionalModule("pymodbus")
  Endian = OptionalModule("pymodbus")
  BinaryPayloadBuilder = OptionalModule("pymodbus")
  BinaryPayloadDecoder = OptionalModule("pymodbus")


REGISTERS: dict[str, tuple[int, int]] = {"Pressure": (1202, 2),
                                         "Temperature": (1204, 2),
                                         "Volumetric_flow": (1206, 2),
                                         "Mass_flow": (1208, 2)}
SETPOINT_REGISTER: int = 1009


class FlowControllerAlicat(InOut):
  """Driver for Alicat mass flow controllers over Modbus RTU.

  .. versionadded:: 2.0.9
  """

  def __init__(self,
               port: str,
               address: int = 1,
               timeout: float = 3.0) -> None:
    """Sets the arguments and initializes parent class.

    Args:
      address: Modbus slave address (unit id).
      port: Serial port on which to communicate with the device (e.g. "COM7" on
        Windows, "/dev/ttyUSB0" on Linux).
      timeout: Read/write timeout (seconds).
    """

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    self._client: ModbusSerialClient | None = None

    super().__init__()

    self._address = address
    self._port = port
    self._timeout = timeout

  def open(self) -> None:
    """Open the serial Modbus RTU connection."""

    self._client = ModbusSerialClient(method="rtu", port=self._port,
                                      baudrate=9600, bytesize=8, parity='N',
                                      stopbits=1, timeout=self._timeout)

    self.log(logging.INFO, f"Connected to Alicat on {self._port}")

  def get_data(self) -> tuple[float, float, float, float, float] | None:
    """Read pressure, temperature, volumetric flow and mass flow from the
    controller in this order."""

    now = time.time()

    data: list[float] = [now]

    for addr, count in REGISTERS:
      resp = self._client.read_holding_registers(address=addr,count=count,
                                                     slave=self._address)

      if resp is None or resp.isError() or not hasattr(resp, "registers"):
        self.log(logging.WARNING, f"Failed to read at address {addr}")
        return None

      decoder = BinaryPayloadDecoder.fromRegisters(resp.registers,
                                                   byteorder=Endian.BIG,
                                                   wordorder=Endian.BIG)
      value = float(decoder.decode_32bit_float())

      data.append(value)

    return data[0], data[1], data[2], data[3], data[4]

  def set_cmd(self, cmd: float | int) -> None:
    """Write the mass flow setpoint (32-bit float)."""

    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
    builder.add_32bit_float(float(cmd))

    registers = builder.to_registers()
    resp = self._client.write_registers(address=SETPOINT_REGISTER,
                                        values=registers, slave=self._address)

    if resp is None or resp.isError():
      raise IOError(f"Alicat write error: {resp}")

    self.log(logging.DEBUG, f"Setpoint command sent: {float(cmd)}")

  def close(self) -> None:
    """Close the Modbus connection."""

    if self._client is None:
      return

    self._client.close()
    self.log(logging.INFO, f"Disconnected from Alicat on {self._port}")
