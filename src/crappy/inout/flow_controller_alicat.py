# coding: utf-8
"""
Alicat Mass Flow Controller interface (Modbus RTU).

This module provides a Crappy InOut driver to communicate with an Alicat
mass flow controller using Modbus RTU over a serial link.

The driver exposes:
- `get_data()` to read process values (Pressure, Temperature, flows...)
- `set_cmd()` to write the mass flow setpoint

Notes:
- Register addresses are based on Alicat documentation and may vary depending
  on model / firmware. Adjust `REGISTERS` if needed.
- This implementation uses pymodbus only.
"""

from __future__ import annotations

import logging
import time
from typing import Final, Optional

from .meta_inout import InOut

try:
  # pymodbus >= 3.x
  from pymodbus.client.serial import ModbusSerialClient
except ModuleNotFoundError:  # pymodbus 2.x
  from pymodbus.client.sync import ModbusSerialClient

from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadBuilder, BinaryPayloadDecoder


REGISTERS: Final[dict[str, tuple[int, int]]] = {
  "Pressure": (1202, 2),
  "Temperature": (1204, 2),
  "Volumetric_flow": (1206, 2),
  "Mass_flow": (1208, 2),
}

SETPOINT_REGISTER: Final[int] = 1009


class FlowControllerAlicat(InOut):
  """Crappy InOut driver for Alicat mass flow controllers over Modbus RTU.

  Args:
      address: Modbus slave address (unit id).
      port: Serial port (e.g. "COM5" on Windows, "/dev/ttyUSB0" on Linux).
      databits: Serial bytesize.
      parity: Serial parity ("N", "E", "O").
      stopbits: Serial stopbits.
      baudrate: Serial baudrate.
      timeout: Read/write timeout (seconds).
      svp: Variables to read, e.g. ["Mass_flow", "Pressure"].
          Values are returned in the same order.

  Returns from get_data():
      list[float]:
          [timestamp, <value for svp[0]>, <value for svp[1]>, ...]
          Missing/invalid values are returned as NaN.
  """

  def __init__(
      self,
      address: int = 1,
      port: str = "/dev/ttyUSB1",
      databits: int = 8,
      parity: str = "N",
      stopbits: int = 1,
      baudrate: int = 9600,
      timeout: float = 3.0,
      svp: Optional[list[str]] = None,
  ) -> None:
    super().__init__()

    self.address = address
    self.port = port
    self.databits = databits
    self.parity = parity
    self.stopbits = stopbits
    self.baudrate = baudrate
    self.timeout = timeout
    self.svp = list(svp) if svp is not None else []

    self._client: Optional[ModbusSerialClient] = None

  def open(self) -> None:
    """Open the serial Modbus RTU connection."""
    self._client = ModbusSerialClient(
        method="rtu",
        port=self.port,
        baudrate=self.baudrate,
        bytesize=self.databits,
        parity=self.parity,
        stopbits=self.stopbits,
        timeout=self.timeout,
    )

    ok = bool(self._client.connect())
    if not ok:
      self.log(logging.ERROR, f"Could not connect to Alicat on {self.port}")
      raise ConnectionError(f"Modbus connection failed on {self.port}")

    self.log(logging.INFO, f"Connected to Alicat on {self.port}")

  def get_data(self) -> list[float]:
    """Read requested variables from the controller."""
    now = time.time()

    if self._client is None:
      return [now] + [float("nan")] * len(self.svp)

    data: list[float] = [now]

    for var in self.svp:
      if var not in REGISTERS:
        self.log(logging.WARNING, f"'{var}' not in supported variables.")
        data.append(float("nan"))
        continue

      addr, count = REGISTERS[var]

      try:
        # pymodbus keyword differs across versions: 'slave' vs 'unit'
        try:
          resp = self._client.read_holding_registers(
              address=addr,
              count=count,
              slave=self.address,
          )
        except TypeError:
          resp = self._client.read_holding_registers(
              address=addr,
              count=count,
              unit=self.address,
          )
      except Exception as exc:
        self.log(logging.ERROR, f"Exception while reading {var}: {exc}")
        data.append(float("nan"))
        continue

      if resp is None or resp.isError() or not hasattr(resp, "registers"):
        self.log(logging.ERROR, f"Failed to read {var} at address {addr}")
        data.append(float("nan"))
        continue

      try:
        decoder = BinaryPayloadDecoder.fromRegisters(
            resp.registers,
            byteorder=Endian.BIG,
            wordorder=Endian.BIG,
        )
        value = float(decoder.decode_32bit_float())
      except Exception as exc:
        self.log(logging.ERROR, f"Decode error for {var}: {exc}")
        value = float("nan")

      data.append(value)

    return data

  def set_cmd(self, cmd: float | int) -> None:
    """Write the mass flow setpoint (32-bit float)."""
    if self._client is None:
      raise ConnectionError(f"Alicat: not connected on {self.port}")

    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
    builder.add_32bit_float(float(cmd))

    # Pymodbus 3.x: builder.to_registers()
    # Older pymodbus: builder.build() -> bytes, then convert to 16-bit registers
    try:
      registers = builder.to_registers()
    except AttributeError:
      payload = builder.build()  # bytes
      if len(payload) % 2 != 0:
        raise ValueError("Invalid payload length for 16-bit registers conversion.")
      registers = [
        (payload[i] << 8) | payload[i + 1]
        for i in range(0, len(payload), 2)
      ]

    try:
      # pymodbus keyword differs across versions: 'slave' vs 'unit'
      try:
        resp = self._client.write_registers(
            address=SETPOINT_REGISTER,
            values=registers,
            slave=self.address,
        )
      except TypeError:
        resp = self._client.write_registers(
            address=SETPOINT_REGISTER,
            values=registers,
            unit=self.address,
        )

      if resp is None or resp.isError():
        raise IOError(f"Alicat write error: {resp}")

    except Exception as exc:
      self.log(logging.ERROR, f"Exception while writing setpoint: {exc}")
      raise

    self.log(logging.INFO, f"Setpoint command sent: {float(cmd)}")

  def close(self) -> None:
    """Close the Modbus connection."""
    if self._client is None:
      return

    try:
      self._client.close()
    finally:
      self._client = None
      self.log(logging.INFO, f"Disconnected from Alicat on {self.port}")
