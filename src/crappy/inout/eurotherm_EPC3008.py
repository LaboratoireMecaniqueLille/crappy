# -*- coding: utf-8 -*-
"""
Eurotherm EPC3008 controller interface (Modbus RTU).

This module provides a Crappy InOut driver to communicate with an Eurotherm
EPC3008 temperature controller using Modbus RTU over a serial link.

The driver exposes:
- `get_data()` to read the process value (PV)
- `set_cmd()` to write the setpoint (SP)

Notes:
- The EPC3008 register map can vary depending on configuration; adjust the
  register addresses to match your controller settings.
- This implementation uses pymodbus only.
"""

from __future__ import annotations

import logging
import time
from typing import Final, Optional

import crappy

try:
    # pymodbus >= 3.x
    from pymodbus.client.serial import ModbusSerialClient
except ModuleNotFoundError:  # pymodbus 2.x
    from pymodbus.client.sync import ModbusSerialClient


REGISTERS: Final[dict[str, tuple[int, int]]] = {
    "process_value": (1, 1),
    "setpoint": (2, 1),
}


class EurothermEPC3008(crappy.inout.InOut):
    """Crappy InOut driver for Eurotherm EPC3008 over Modbus RTU (serial).

    Args:
        port: Serial port (e.g. "COM5" on Windows, "/dev/ttyUSB0" on Linux).
        address: Modbus slave address (unit id).
        baudrate: Serial baudrate.
        bytesize: Serial bytesize.
        parity: Serial parity ("N", "E", "O").
        stopbits: Serial stopbits.
        timeout: Read/write timeout (seconds).
        registers: Optional register map overriding `REGISTERS`.
            Expected keys: "process_value", "setpoint".
            Values: (address, length).

    Returns from get_data():
        dict: {"t(s)": <timestamp>, "T(C)": <pv>}
        Where "T(C)" is None if the value could not be read.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        address: int = 1,
        baudrate: int = 9600,
        bytesize: int = 8,
        parity: str = "N",
        stopbits: int = 1,
        timeout: float = 1.0,
        registers: Optional[dict[str, tuple[int, int]]] = None,
    ) -> None:
        super().__init__()

        self.port = port
        self.address = address
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.timeout = timeout

        self.registers: dict[str, tuple[int, int]] = (
            dict(registers) if registers is not None else dict(REGISTERS)
        )

        self._client: Optional[ModbusSerialClient] = None

    def open(self) -> None:
        """Open the serial Modbus RTU connection."""
        self._client = ModbusSerialClient(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=self.bytesize,
            parity=self.parity,
            stopbits=self.stopbits,
            timeout=self.timeout,
            method="rtu",
        )

        ok = bool(self._client.connect())
        if not ok:
            self.log(logging.ERROR, f"Eurotherm EPC3008: cannot connect on {self.port}")
            raise ConnectionError(f"Could not connect to Eurotherm EPC3008 on {self.port}")

        self.log(logging.INFO, f"Eurotherm EPC3008 connected on {self.port}")

    def get_data(self) -> dict[str, float | None]:
        """Read process value (PV) and setpoint (SP)."""
        now = time.time()
    
        if self._client is None:
            return {"t(s)": now, "T(C)": None, "T_cons(C)": None}

        try:
            # PV
            pv_addr, pv_len = self.registers["process_value"]
            sp_addr, sp_len = self.registers["setpoint"]
    
            try:
                pv_resp = self._client.read_holding_registers(
                    address=pv_addr, count=pv_len, slave=self.address
                )
                sp_resp = self._client.read_holding_registers(
                    address=sp_addr, count=sp_len, slave=self.address
                )
            except TypeError:
                pv_resp = self._client.read_holding_registers(
                    address=pv_addr, count=pv_len, unit=self.address
                )
                sp_resp = self._client.read_holding_registers(
                    address=sp_addr, count=sp_len, unit=self.address
                )

            if (
                pv_resp.isError()
                or sp_resp.isError()
                or not hasattr(pv_resp, "registers")
                or not hasattr(sp_resp, "registers")
            ):
                self.log(logging.ERROR, "Eurotherm read error")
                return {"t(s)": now, "T(C)": None, "T_cons(C)": None}

            pv = float(pv_resp.registers[0])
            sp = float(sp_resp.registers[0])

            return {"t(s)": now, "T(C)": pv, "T_cons(C)": sp}

        except Exception as exc:
            self.log(logging.ERROR, f"Eurotherm exception while reading: {exc}")
            return {"t(s)": now, "T(C)": None, "T_cons(C)": None}

    def set_cmd(self, cmd: float | int) -> None:
        """Write the setpoint (SP)."""
        if self._client is None:
            raise ConnectionError(f"Eurotherm EPC3008: not connected on {self.port}")

        reg_addr, _ = self.registers["setpoint"]
        value = int(cmd) & 0xFFFF

        try:
            # pymodbus keyword differs across versions: 'slave' vs 'unit'
            try:
                resp = self._client.write_register(
                    address=reg_addr,
                    value=value,
                    slave=self.address,
                )
            except TypeError:
                resp = self._client.write_register(
                    address=reg_addr,
                    value=value,
                    unit=self.address,
                )

            if resp.isError():
                raise IOError(f"Eurotherm EPC3008 write error: {resp}")

        except Exception as exc:
            self.log(logging.ERROR, f"Eurotherm EPC3008 exception while writing: {exc}")
            raise

    def close(self) -> None:
        """Close the serial Modbus RTU connection."""
        if self._client is None:
            return

        try:
            self._client.close()
        finally:
            self._client = None
