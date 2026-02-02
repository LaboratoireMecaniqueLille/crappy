# coding: utf-8

import time
import logging
from  warnings import warn

from .meta_inout import InOut
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")

# Addresses of the registers of the device
CMD_READ = 0x52
CMD_WRITE = 0x43
PARAM_PROGRAM = 0x15
PARAM_SV_MV = 0x1A
STOP_MODE_PARAM = 12


class SagerSG_GS1700(InOut):
  """Driver for Sager SG-GS1700 furnace via AIBUS (serial).

  .. versionadded:: 2.0.9
  """

  def __init__(self,
               port: str,
               baudrate: int = 9600,
               timeout: float = 0.8,
               address: int = 0x0A) -> None:
    """Sets the arguments and initializes parent class.

    Args:
      port: Serial port on which to communicate with the device (e.g. "COM7" on
        Windows, "/dev/ttyUSB0" on Linux).
      baudrate: Serial baudrate.
      timeout: Serial timeout (seconds).
      address: AIBUS device address.
    """

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    self._ser: serial.Serial | None = None

    super().__init__()

    self._compensations_dict = dict()
    self._port = port
    self._baudrate = baudrate
    self._timeout = timeout
    self._address = address

  def open(self) -> None:
    """Open the serial connection and put the furnace in hold mode."""

    self._ser = serial.Serial(self._port, self._baudrate, bytesize=8,
                              parity=serial.PARITY_NONE, stopbits=1,
                              timeout=self._timeout)
    self.log(logging.INFO, f"Sager SG-GS1700 connected on {self._port}")

    self._write_param_u16(PARAM_PROGRAM, 5)

  def get_data(self) -> tuple[float, float, float] | None:
    """Read PV and SV (in °C)."""

    now = time.time()

    # Cannot interpret the read values in that case
    if len(resp := self._read_generic(0x00)) < 4:
      return None

    pv_u16 = (resp[1] << 8) | resp[0]
    sv_u16 = (resp[3] << 8) | resp[2]

    return now, pv_u16 / 10.0, sv_u16 / 10.0

  def set_cmd(self, value: float | int) -> None:
    """Write the temperature setpoint (°C)."""

    # Put the furnace in hold mode
    self._write_param_u16(PARAM_PROGRAM, 5)

    val = int(round(float(value) * 10.0)) & 0xFFFF
    self._write_param_u16(PARAM_SV_MV, val)

  def close(self) -> None:
    """Close the serial connection."""

    # Only close if the serial connection was first opened
    if self._ser is None:
      return

    # Put the furnace in stop mode
    self._write_param_u16(STOP_MODE_PARAM, 10)

    self._ser.close()
    self.log(logging.INFO, f"Sager SG-GS1700 disconnected from {self._port}")

  def _build_frame(self, cmd: int, param: int, value_u16: int) -> bytes:
    """Build an AIBUS frame from a command and a value."""

    val_lo = value_u16 & 0xFF
    val_hi = (value_u16 >> 8) & 0xFF

    # Compute checksum
    ecc = ((param << 8) + cmd + value_u16 + self._address) & 0xFFFF
    ecc_lo = ecc & 0xFF
    ecc_hi = (ecc >> 8) & 0xFF

    return bytes([0x8A, 0x8A, cmd, param, val_lo, val_hi, ecc_lo, ecc_hi])

  def _send_and_recv(self, frame: bytes, read_len: int = 12) -> bytes:
    """Send a frame and read the response."""

    self._ser.reset_input_buffer()
    self._ser.reset_output_buffer()

    self.log(logging.DEBUG, f"TX: {' '.join(f'{b:02X}' for b in frame)}")
    self._ser.write(frame)
    self._ser.flush()

    time.sleep(0.08)

    resp = self._ser.read(read_len)
    self.log(logging.DEBUG, f"RX: {' '.join(f'{b:02X}' for b in resp)}")

    return resp

  def _write_param_u16(self, param: int, value: int) -> None:
    """Write a 16-bit parameter."""

    frame = self._build_frame(CMD_WRITE, param, value & 0xFFFF)
    self._send_and_recv(frame, read_len=8)

  def _read_generic(self, param: int = 0x00) -> bytes:
    """Generic read command for a given param."""

    frame = self._build_frame(CMD_READ, param, 0x0000)
    return self._send_and_recv(frame, read_len=12)
