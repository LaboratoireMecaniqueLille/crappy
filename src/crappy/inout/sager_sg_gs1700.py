# -*- coding: utf-8 -*-
"""
Sager SG-GS1700 furnace interface (AIBUS over serial).

This module provides a Crappy InOut driver to communicate with a Sager
SG-GS1700 furnace controller using the AIBUS protocol over a serial link.

The driver exposes:
- get_data(): read PV (process value) and SV (setpoint/consigne)
- set_cmd(): write the temperature setpoint (SV)

Notes:
- AIBUS frame format and ECC computation follow the user's existing implementation.
- Register/parameter meanings depend on the controller configuration.
"""

from __future__ import annotations

import time
import logging
from typing import Optional

import crappy
import serial


class SagerSG_GS1700(crappy.inout.InOut):
    """Crappy InOut driver for Sager SG-GS1700 furnace via AIBUS (serial).

    Args:
        port: Serial port (e.g. "COM7" on Windows, "/dev/ttyUSB0" on Linux).
        baudrate: Serial baudrate.
        timeout: Serial timeout (seconds).
        address: AIBUS device address.
        debug: If True, print TX/RX frames in hex.

    Returns from get_data():
        dict: {"t(s)": <timestamp>, "T(C)": <pv>, "T_cons(C)": <sv>}
        Values are None if unavailable.
    """

    _CMD_READ = 0x52
    _CMD_WRITE = 0x43
    _PARAM_PROGRAM = 0x15
    _PARAM_SV_MV = 0x1A
    STOP_MODE_PARAM = 12

    def __init__(
        self,
        port: str = "COM7",
        baudrate: int = 9600,
        timeout: float = 0.8,
        address: int = 0x0A,
        debug: bool = True,
    ) -> None:
        super().__init__()
        self._compensations_dict = {}

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.address = address
        self.debug = debug

        self._ser: Optional[serial.Serial] = None

    # ------------------------------------------------------------------
    # Low-level helpers (AIBUS)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_u16_le(lo: int, hi: int) -> int:
        """Combine 2 bytes (little-endian) into an unsigned 16-bit integer."""
        return (hi << 8) | lo

    def _log_hex(self, prefix: str, data: bytes) -> None:
        """Print frame bytes in hex if debug is enabled."""
        if self.debug:
            print(f"{prefix}: {' '.join(f'{b:02X}' for b in data)}")

    def _calc_ecc(self, cmd: int, param: int, value_u16: int) -> int:
        """Compute ECC (checksum) as implemented in the original script."""
        return ((param << 8) + cmd + value_u16 + self.address) & 0xFFFF

    def _build_frame(self, cmd: int, param: int, value_u16: int) -> bytes:
        """Build an AIBUS frame."""
        val_lo = value_u16 & 0xFF
        val_hi = (value_u16 >> 8) & 0xFF

        ecc = self._calc_ecc(cmd, param, value_u16)
        ecc_lo = ecc & 0xFF
        ecc_hi = (ecc >> 8) & 0xFF

        return bytes([0x8A, 0x8A, cmd, param, val_lo, val_hi, ecc_lo, ecc_hi])

    def _send_and_recv(self, frame: bytes, read_len: int = 12) -> bytes:
        """Send a frame and read the response."""
        if self._ser is None:
            raise ConnectionError("Serial port not open. Call open() first.")

        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

        self._log_hex("TX", frame)
        self._ser.write(frame)
        self._ser.flush()

        time.sleep(0.08)

        resp = self._ser.read(read_len)
        self._log_hex("RX", resp)
        return resp

    def _write_param_u16(self, param: int, value: int) -> bool:
        """Write a 16-bit parameter."""
        frame = self._build_frame(self._CMD_WRITE, param, value & 0xFFFF)
        resp = self._send_and_recv(frame, read_len=8)
        return len(resp) > 0

    def _read_generic(self, param: int = 0x00) -> bytes:
        """Generic read command for a given param."""
        frame = self._build_frame(self._CMD_READ, param, 0x0000)
        return self._send_and_recv(frame, read_len=12)

    # ------------------------------------------------------------------
    # Crappy API
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the serial connection."""
        self._ser = serial.Serial(
            self.port,
            self.baudrate,
            bytesize=8,
            parity=serial.PARITY_NONE,
            stopbits=1,
            timeout=self.timeout,
        )
        self.log(logging.INFO, f"Sager SG-GS1700 connected on {self.port}")

    def begin(self) -> None:
        """Put the furnace in HOLD mode (as in the original script)."""
        if self._ser is None:
            raise ConnectionError("Sager not connected. Call open() first.")

        self._write_param_u16(self._PARAM_PROGRAM, 5)

    def get_data(self) -> dict[str, float | None]:
        """Read PV and SV (in °C)."""
        now = time.time()

        try:
            resp = self._read_generic(0x00)
        except Exception as exc:
            self.log(logging.ERROR, f"Sager read exception: {exc}")
            return {"t(s)": now, "T(C)": None, "T_cons(C)": None}

        if len(resp) < 4:
            return {"t(s)": now, "T(C)": None, "T_cons(C)": None}

        pv_u16 = self._to_u16_le(resp[0], resp[1])
        sv_u16 = self._to_u16_le(resp[2], resp[3])

        return {
            "t(s)": now,
            "T(C)": pv_u16 / 10.0,
            "T_cons(C)": sv_u16 / 10.0,
        }

    def set_cmd(self, value: float | int) -> None:
        """Write the temperature setpoint (°C)."""
        if self._ser is None:
            raise ConnectionError("Sager not connected. Call open() first.")

        self._write_param_u16(self._PARAM_PROGRAM, 5)

        val = int(round(float(value) * 10.0)) & 0xFFFF
        self._write_param_u16(self._PARAM_SV_MV, val)

    def finish(self) -> None:
        """Finish sequence (as in the original script)."""
        if self._ser is None:
            raise ConnectionError("Sager not connected. Call open() first.")

        self._write_param_u16(self.STOP_MODE_PARAM, 10)

    def close(self) -> None:
        """Close the serial connection."""
        if self._ser is None:
            return

        try:
            self._ser.close()
        finally:
            self._ser = None
            self.log(logging.INFO, f"Sager SG-GS1700 disconnected from {self.port}")
