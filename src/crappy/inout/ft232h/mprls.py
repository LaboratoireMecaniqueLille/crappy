# coding: utf-8

from time import time
from typing import Optional
import logging

from ..meta_inout import InOut
from ...tool.ft232h import FT232HServer as FT232H, I2CMessage, USBArgsType

mprls_status_bits = {'busy': 0x20,
                     'memory error': 0x04,
                     'math saturation': 0x01}


class MPRLSFT232H(InOut):
  """This class can read values from an MPRLS pressure sensor through an
  FT232H.

  It is similar to the :class:`~crappy.inout.MPRLS` class, except this class is
  specific for use with an :class:`~crappy.tool.ft232h.FT232H` USB to I2C
  converter.

  It communicates over I2C with the sensor.

  .. versionadded:: 2.0.0
  """

  ft232h = True

  def __init__(self,
               eoc_pin: Optional[str] = None,
               device_address: int = 0x18,
               _ft232h_args: USBArgsType = tuple()) -> None:
    """Initializes the parent class and opens the I2C bus.

    Args:
      eoc_pin: Optionally, reads the end of conversion signal from the polarity
        of a GPIO rather than from an I2C register. Speeds up the reading and
        decreases the traffic on the bus, but requires one extra wire. Give the
        name of the GPIO in the format `Dx` or `Cx`.
      device_address: The I2C address of the MPRLS. The address of the devices
        sold by Adafruit is `0x18`, but other suppliers may sell it with
        another address.
      _ft232h_args: This argument is meant for internal use only and should not
        be provided by the user. It contains the information necessary for
        setting up the FT232H.
    """

    self._bus = None

    super().__init__()

    (block_index, block_lock, command_file, answer_file, shared_lock,
     current_block) = _ft232h_args

    self._bus = FT232H(mode='I2C',
                       block_index=block_index,
                       current_block=current_block,
                       command_file=command_file,
                       answer_file=answer_file,
                       block_lock=block_lock,
                       shared_lock=shared_lock)

    if not isinstance(device_address, int):
      raise TypeError("device_address should be an integer.")
    self._address = device_address

    if eoc_pin is not None and not isinstance(eoc_pin, str):
      raise TypeError('eoc_pin should be a string when using the ft232h '
                      'backend !')
    self._eoc_pin = eoc_pin

    self._i2c_msg = None

  def open(self) -> None:
    """Opens the I2C bus."""

    self._i2c_msg = I2CMessage

  def get_data(self) -> list[float]:
    """Reads the pressure value.

    Returns:
      The timestamp and the pressure value in hPa.
    """

    # Starting conversion
    self.log(logging.DEBUG, f"Writing {0xAA, 0x00, 0x00} to the address "
                            f"{self._address}")
    self._bus.i2c_rdwr(self._i2c_msg.write(self._address,
                                           [0xAA, 0x00, 0x00]))
    # Waiting for conversion to complete
    t0 = time()
    while not self._data_available():
      if time() - t0 > 0.1:
        raise TimeoutError('Waited too long for data to be ready')

    # Reading conversion result
    read = self._i2c_msg.read(self._address, 4)
    self._bus.i2c_rdwr(read)
    out = list(read)
    self.log(logging.DEBUG, f"Read {out} from address {self._address}")
    # Checking if anu error occurred
    if out[0] & mprls_status_bits['memory error']:
      raise RuntimeError("A memory error occurred on the MPRLS")
    elif out[0] & mprls_status_bits['math saturation']:
      raise RuntimeError("A math saturation error occurred on the MPRLS")
    # Extracting conversion result as an integer
    ret = (out[1] << 16) | (out[2] << 8) | out[3]
    # Converting to hPa
    pres = 68.947572932 * (ret - 0x19999A) * 25 / (0xE66666 - 0x19999A)

    return [time(), pres]

  def close(self) -> None:
    """Closes the I2C bus."""

    if self._bus is not None:
      self.log(logging.INFO, "Closing the I2C connection to the MPRLS")
      self._bus.close()

  def _data_available(self) -> bool:
    """Returns :obj:`True` if data is available, :obj:`False` otherwise."""

    # EOC signal from the I2C communication
    if self._eoc_pin is None:
      wait = self._i2c_msg.read(self._address, 1)
      self._bus.i2c_rdwr(wait)
      return not list(wait)[0] & mprls_status_bits['busy']
    # EOC signal from a GPIO
    else:
      return bool(self._bus.get_gpio(self._eoc_pin))
