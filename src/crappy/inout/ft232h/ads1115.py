# coding: utf-8

from time import time
from typing import Optional, Literal
import logging

from ..meta_inout import InOut
from ...tool.ft232h import FT232HServer as FT232H, USBArgsType

# Register and other configuration values:
Ads1115_pointer_conversion = 0x00
Ads1115_pointer_config = 0x01
Ads1115_pointer_lo_thresh = 0x02
Ads1115_pointer_hi_thresh = 0x03

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


class ADS1115FT232H(InOut):
  """A class for controlling Adafruit's ADS1115 16-bits ADC through an FT232H.

  It is similar to the :class:`~crappy.inout.ADS1115` class, except this class
  is specific for use with an :class:`~crappy.tool.ft232h.FT232H` USB to I2C
  converter.

  The ADS1115 InOut is meant for reading conversion values from a 16-bits
  ADS1115 ADC, using the I2C protocol. The output is in Volts by default, but a
  ``gain`` and an ``offset`` can be specified.

  Various settings can be adjusted, like the sample rate, the input mode or the
  voltage range.
  
  .. versionadded:: 2.0.0
  """

  ft232h = True

  def __init__(self,
               device_address: int = 0x48,
               sample_rate: int = 128,
               v_range: float = 2.048,
               multiplexer: Literal['A0', 'A1', 'A2', 'A3', 'A0 - A1',
                                    'A0 - A3', 'A1 - A3', 'A2 - A3'] = 'A1',
               dry_pin: Optional[str] = None,
               gain: float = 1,
               offset: float = 0,
               _ft232h_args: USBArgsType = tuple()) -> None:
    """Checks the validity of the arguments.

    Args:
      device_address: The I2C address of the ADS1115. The default address is
        `0x48`, but it is possible to change this setting using the `ADDR` pin.
      sample_rate: The sample rate for data conversion (in SPS). The available
        sample rates are:
        ::

          8, 16, 32, 64, 128, 250, 475, 860

      v_range: The value (in Volts) of the measured signal corresponding to the
        `0x7FFF` output in bits, i.e. that saturates the sensor. A signal of
        ``-v_range`` Volts gives a `0x8000` output in bits. Available
        ``v_range`` values are:
        ::

          0.256, 0.512, 1.024, 2.048, 4.096, 6.144

      multiplexer: Choice of the inputs to consider. Single-input modes
        actually measure `Ax - GND`. The available ``multiplexer`` values are:
        ::

          'A0', 'A1', 'A2', 'A3',
          'A0 - A1',
          'A0 - A3',
          'A1 - A3',
          'A2 - A3'

      dry_pin: Optionally, reads the end of conversion signal from a GPIO
        rather than from an I2C message. Speeds up the reading and decreases
        the traffic on the I2C bus, but requires one extra wire. Give the name
        of the GPIO in the format `Dx` or `Cx`.
      gain: Allows to tune the output value according to the formula :
        :math:`output = gain * tension + offset`.
      offset: Allows to tune the output value according to the formula :
        :math:`output = gain * tension + offset`.
      _ft232h_args: This argument is meant for internal use only and should not
        be provided by the user. It contains the information necessary for
        setting up the FT232H.

    Warning:
      AINx voltages should not be higher than `VDD+0.3V` nor lower than
      `GND-0.3V`. Setting high ``v_range`` values does not allow measuring
      voltages higher than `VDD` !!
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

    if dry_pin is not None and not isinstance(dry_pin, str):
      raise TypeError('int_pin should be a string !')
    self._dry_pin = dry_pin

    self._gain = gain
    self._offset = offset

  def open(self) -> None:
    """Initializes the I2C communication and the device."""

    if not self._is_connected():
      raise IOError("The ADS1115 is not connected")

    # Setting the configuration register according to the user values
    init_value = Ads1115_config_mux[self._multiplexer]
    init_value |= Ads1115_config_gain[self._v_range]
    init_value |= 0x0100  # Single shot operating mode
    init_value |= Ads1115_config_dr[self._sample_rate]
    self.log(logging.INFO, "Initializing the ADS1115")
    if self._dry_pin is None:
      # Setting the two last bits to 11 to disable the DRY pin
      init_value |= 0x0003

    self._set_register(Ads1115_pointer_config, init_value)

    # Setting the threshold registers to activate the DRY pin output
    if self._dry_pin is not None:
      self._set_register(Ads1115_pointer_lo_thresh, 0x0000)
      self._set_register(Ads1115_pointer_hi_thresh, 0xFFFF)

  def get_data(self) -> list[float]:
    """Reads the registers containing the conversion result.

    The output is in Volts, unless a gain and offset are applied.

    Returns:
      A :obj:`list` containing the timestamp and the voltage value.
    """

    # Reading the config register, and setting it so that the ADS1115 starts
    # a conversion
    ms_byte, ls_byte = self._bus.read_i2c_block_data(self._device_address,
                                                     Ads1115_pointer_config,
                                                     2)
    self._set_register(Ads1115_pointer_config,
                       ((ms_byte | 0x80) << 8) | ls_byte)

    # Waiting for the end of the conversion
    t0 = time()
    while not self._data_available():
      if time() - t0 > 0.5:
        raise TimeoutError('Waited too long for data to be ready !')

    # Reading the output of the conversion
    out = [time()]
    ms_byte, ls_byte = self._bus.read_i2c_block_data(
      self._device_address,
      Ads1115_pointer_conversion,
      2)
    self.log(logging.DEBUG, f"Read {ms_byte, ls_byte} from the device "
                            f"address {self._device_address} at "
                            f"register {Ads1115_pointer_conversion}")

    # Converting the output value into Volts
    value_raw = (ms_byte << 8) | ls_byte
    if ms_byte >> 7:
      value_raw -= 2 ** 16
    value = value_raw * self._v_range / 2 ** 15

    out.append(self._offset + self._gain * value)
    return out

  def close(self) -> None:
    """Closes the I2C bus."""

    if self._bus is not None:
      self.log(logging.INFO, "Closing the ADS1115")
      self._bus.close()

  def _set_register(self, register_address: int, value: int) -> None:
    """Thin wrapper for writing data to the registers."""

    self.log(logging.DEBUG, f"Writing {[(value >> 8) & 0xFF, value & 0xFF]} "
                            f"to the address {self._device_address} in "
                            f"register {register_address}")
    self._bus.write_i2c_block_data(self._device_address, register_address,
                                   [(value >> 8) & 0xFF, value & 0xFF])

  def _is_connected(self) -> bool:
    """Tries reading a byte from the device.

    Returns:
      :obj:`True` if reading was successful, else :obj:`False`.
    """

    try:
      self._bus.read_byte(self._device_address)
      return True
    except IOError:
      return False

  def _data_available(self) -> bool:
    """Returns :obj:`True` if data is available, :obj:`False` otherwise."""

    # EOC signal from the I2C communication
    if self._dry_pin is None:
      return bool(self._bus.read_i2c_block_data(self._device_address,
                                                Ads1115_pointer_config,
                                                1)[0] & 0x80)
    # EOC signal from a GPIO
    else:
      return not bool(self._bus.get_gpio(self._dry_pin))
