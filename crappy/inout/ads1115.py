# coding: utf-8

import time
from .inout import InOut
from ..tool import ft232h
from .._global import OptionalModule

try:
  import smbus2
except (ModuleNotFoundError, ImportError):
  smbus2 = OptionalModule("smbus2")

# Register and other configuration values:
Ads1115_default_address = 0x48
Ads1115_pointer_conversion = 0x00
Ads1115_pointer_config = 0x01

Ads1115_config_mux = {'A0 - A1': 0x0000,
                      'A0 - A3': 0x1000,
                      'A1 - A3': 0x2000,
                      'A2 - A3': 0x3000,
                      'A0': 0x4000,
                      'A1': 0x5000,
                      'A2': 0x6000,
                      'A3': 0x7000}

Ads1115_config_gain = {0.256: 0x0a00,
                       0.512: 0x0800,
                       1.024: 0x0600,
                       2.048: 0x0400,
                       4.096: 0x0200,
                       6.144: 0x0000}

Ads1115_config_dr = {8: 0x0000,
                     16: 0x0020,
                     32: 0x0040,
                     64: 0x0060,
                     128: 0x0080,
                     250: 0x00A0,
                     475: 0x00C0,
                     860: 0x00E0}

Ads1115_backends = ['Pi4', 'ft232h']


class Ads1115(InOut):
  """A class for controlling Adafruit's ADS1115 16-bits ADC.

  The Ads1115 InOut block is meant for reading output values from a 16-bits
  ADS1115 ADC, using the I2C protocol. The output is in Volts by default, but a
  ``gain`` and an ``offset`` can be specified.

  Warning:
    Only available on Raspberry Pi for now !
  """

  def __init__(self,
               backend: str,
               device_address: int = Ads1115_default_address,
               i2c_port: int = 1,
               sample_rate: int = 128,
               v_range: float = 2.048,
               multiplexer: str = 'A1',
               gain: float = 1,
               offset: float = 0,
               ft232h_ser_num: str = None) -> None:
    """Checks arguments validity.

    Args:
      backend (:obj:`str`): The backend for communicating with the NAU7802.
        Should be one of:
        ::

          'Pi4', 'ft232h'

      device_address (:obj:`int`, optional): The I2C address of the ADS1115.
        The default address is `0x48`, but it is possible to change this
        setting using the `ADDR` pin.
      i2c_port (:obj:`int`, optional): The I2C port over which the ADS1115
        should communicate. On most Raspberry Pi models the default I2C port is
        `1`.
      sample_rate (:obj:`int`, optional): The sample rate for data conversion
        (in SPS). Available sample rates are:
        ::

          8, 16, 32, 64, 128, 250, 475, 860

      v_range (:obj:`float`, optional): The value (in Volts) of the measured
        signal corresponding to the `0x7FFF` output in bits. A signal of
        ``-v_range`` Volts gives a `0x8000` output in bits. Available
        ``v_range`` values are:
        ::

          0.256, 0.512, 1.024, 2.048, 4.096, 6.144

      multiplexer (:obj:`str`, optional): Choice of the inputs to consider.
        Single-input modes actually measure `Ax - GND`. The available
        ``multiplexer`` values are:
        ::

          'A0', 'A1', 'A2', 'A3',
          'A0 - A1',
          'A0 - A3',
          'A1 - A3',
          'A2 - A3'

      gain (:obj:`float`, optional): Allows to tune the output value according
        to the formula:
        ::

          output = gain * tension + offset.

      offset (:obj:`float`, optional): Allows to tune the output value
        according to the formula:
        ::

          output = gain * tension + offset.

      ft232h_ser_num (:obj:`str`, optional): If backend is `'ft232h'`, the
        serial number of the ft232h to use for communication.

    Warning:
      AINx voltages should not be higher than `VDD+0.3V` nor lower than
      `GND-0.3V`. Setting high ``v_range`` values does not allow to measure
      voltages higher than `VDD` !!
    """

    InOut.__init__(self)
    if backend not in Ads1115_backends:
      raise ValueError("backend should be in {}".format(Ads1115_backends))

    if backend == 'Pi4':
      self._bus = smbus2.SMBus(i2c_port)
    else:
      self._bus = ft232h('I2C', ft232h_ser_num)
    self._device_address = device_address

    if v_range not in Ads1115_config_gain:
      raise ValueError("v_range should be in {}".format(list(
        Ads1115_config_gain.keys())))
    else:
      self._v_range = v_range

    if sample_rate not in Ads1115_config_dr:
      raise ValueError("sample_rate should be in {}".format(list(
        Ads1115_config_dr.keys())))
    else:
      self._sample_rate = sample_rate

    if multiplexer not in Ads1115_config_mux:
      raise ValueError("multiplexer should be in {}".format(list(
        Ads1115_config_mux.keys())))
    else:
      self._multiplexer = multiplexer

    self._gain = gain
    self._offset = offset

  def open(self) -> None:
    """Sets the I2C communication and device."""

    if not self._is_connected():
      raise IOError("The ADS1115 is not connected")

    # Setting the configuration register according to the user values
    init_value = 0x0000
    init_value |= Ads1115_config_mux[self._multiplexer]
    init_value |= Ads1115_config_gain[self._v_range]
    init_value |= 0x0100  # Single shot operating mode
    init_value |= Ads1115_config_dr[self._sample_rate]
    init_value |= 0x0003  # set remaining parameters to default value

    self._set_register(Ads1115_pointer_config, init_value)

  def get_data(self) -> list:
    """Reads the registers containing the conversion result.

    The output is in Volts, unless a gain and offset are applied.

    Returns:
      :obj:`list`: A list containing the timeframe and the voltage value
    """

    # Reading the config register, and setting it so that the ADS1115 starts
    # a conversion
    ms_byte, ls_byte = self._bus.read_i2c_block_data(self._device_address,
                                                     Ads1115_pointer_config, 2)
    ms_byte |= (1 << 7)
    self._set_register(Ads1115_pointer_config, (ms_byte << 8) | ls_byte)

    # Reading the output of the conversion
    time.sleep(1 / self._sample_rate + 0.00005)
    out = [time.time()]
    ms_byte, ls_byte = self._bus.read_i2c_block_data(self._device_address,
                                                     Ads1115_pointer_conversion,
                                                     2)

    # Converting the output value into Volts
    if ms_byte >> 7:
      value = (-(2 ** 15 - 1) + (((ms_byte << 8) | ls_byte) & 0x7fff)) \
              * self._v_range / (2 ** 15 - 1)
    else:
      value = ((ms_byte << 8) | ls_byte) * self._v_range / (2 ** 15 - 1)
    out.append(self._offset + self._gain * value)
    return out

  def close(self) -> None:
    """Closes the I2C bus"""

    self._bus.close()

  def _set_register(self, register_address: int, value: int) -> None:
    """Thin wrapper for writing data to the registers."""

    ms_byte = value >> 8 & 0xff
    ls_byte = value & 0xff
    self._bus.write_i2c_block_data(self._device_address, register_address,
                                   [ms_byte, ls_byte])

  def _is_connected(self) -> bool:
    """Tries reading a byte from the device.

    Returns:
      :obj:`bool`: :obj:`True` if reading was successful, else :obj:`False`
    """

    try:
      self._bus.read_byte(self._device_address)
      return True
    except IOError:
      return False
