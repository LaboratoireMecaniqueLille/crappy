# coding: utf-8

from time import time
from typing import Literal
import logging
from  warnings import warn

from ..meta_inout import InOut
from ...tool.ft232h import FT232HServer as FT232H, USBArgsType

Mcp9600_registers = {'Hot Junction Temperature': 0x00,
                     'Junction Temperature Delta': 0x01,
                     'Cold Junction Temperature': 0x02,
                     'Raw Data ADC': 0x03,
                     'Status': 0x04,
                     'Thermocouple Sensor Configuration': 0x05,
                     'Device Configuration': 0x06,
                     'Alert 1 Configuration': 0x08,
                     'Alert 2 Configuration': 0x09,
                     'Alert 3 Configuration': 0x0A,
                     'Alert 4 Configuration': 0x0B,
                     'Alert 1 Hysteresis': 0x0C,
                     'Alert 2 Hysteresis': 0x0D,
                     'Alert 3 Hysteresis': 0x0E,
                     'Alert 4 Hysteresis': 0x0F,
                     'Alert 1 Limit': 0x10,
                     'Alert 2 Limit': 0x11,
                     'Alert 3 Limit': 0x12,
                     'Alert 4 Limit': 0x13,
                     'Device ID/REV': 0x20}

Mcp9600_filter_coefficients = {0: 0b000,
                               1: 0b001,
                               2: 0b010,
                               3: 0b011,
                               4: 0b100,
                               5: 0b101,
                               6: 0b110,
                               7: 0b111}

Mcp9600_thermocouple_types = {'J': 0b0000000,
                              'K': 0b0010000,
                              'T': 0b0100000,
                              'N': 0b0110000,
                              'S': 0b1000000,
                              'E': 0b1010000,
                              'B': 0b1100000,
                              'R': 0b1110000}

Mcp9600_adc_resolutions = {18: 0b0000000,
                           16: 0b0100000,
                           14: 0b1000000,
                           12: 0b1100000}

Mcp9600_sensor_resolutions = {0.0625: 0b00000000,
                              0.25: 0b10000000}

Mcp9600_modes = ['Hot Junction Temperature',
                 'Junction Temperature Delta',
                 'Cold Junction Temperature',
                 'Raw Data ADC']


class MCP9600FT232H(InOut):
  """This class can read temperature values from an MCP9600 thermocouple
  reader through an FT232H.

  It is similar to the :class:`~crappy.inout.MCP9600` class, except this class
  is specific for use with an :class:`~crappy.tool.ft232h.FT232H` USB to I2C
  converter.

  It communicates over the I2C protocol. The output is in `°C`, except for one
  operating mode that returns Volts. Several parameters can be tuned, like the
  thermocouple type, the reading resolution or the filter coefficient. Note
  that the MCP9600 can only achieve a data rate of a few Hz.

  .. versionadded:: 2.0.0
  """

  ft232h = True

  def __init__(self,
               thermocouple_type: Literal['J', 'K', 'T', 'N', 'S',
                                          'E', 'B', 'R'],
               device_address: int = 0x67,
               adc_resolution: int = 18,
               sensor_resolution: float = 0.0625,
               filter_coefficient: int = 0,
               mode: Literal['Hot Junction Temperature',
                             'Junction Temperature Delta',
                             'Cold Junction Temperature',
                             'Raw Data ADC'] = 'Hot Junction Temperature',
               _ft232h_args: USBArgsType = tuple()) -> None:
    """Checks the validity of the arguments.

    Args:
      thermocouple_type: The type of thermocouple connected to the MCP9600. The
        possible types are:
        ::

          'J', 'K', 'T', 'N', 'S', 'E', 'B', 'R'

      device_address: The I2C address of the MCP9600. The default address is
        `0x67`, but it is possible to change this setting using a specific
        setup involving the `ADDR` pin.
      adc_resolution: The number of bits the ADC output is encoded on. The
        greater the resolution, the lower the sample rate. The available
        resolutions are:
        ::

          12, 14, 16, 18

      sensor_resolution: The temperature measurement resolution in `°C`. It
        should be either `0.0625` or `0.25`. Setting the resolution to `0.25`
        will increase the sample rate, but the output temperature will be
        encoded on two bits less.
      filter_coefficient: The MCP9600 features an integrated filter (see its
        documentation for the exact filter formula). When set to `0`, the
        filter is inactive. It is maximal when set to `7`. When active, the
        filter will prohibit fast temperature changes, thus limiting noise and
        smoothening the signal.
      mode: Four different values can be accessed when measuring a temperature:
        the temperature of the thermocouple (hot junction temperature), the
        temperature of the MCP9600 board (cold junction temperature), the
        temperature calculated from the ADC data and thermocouple type but not
        yet cold junction-compensated (junction temperature delta), and the raw
        ADC measurement of the voltage difference in the thermocouple (raw data
        ADC, in Volts). The available modes are thus:
        ::

          'Hot Junction Temperature',
          'Junction Temperature Delta',
          'Cold Junction Temperature',
          'Raw Data ADC'

      _ft232h_args: This argument is meant for internal use only and should not
        be provided by the user. It contains the information necessary for
        setting up the FT232H.
    """

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

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
                       shared_lock=shared_lock,
                       i2c_speed=20E3)

    self._device_address = device_address

    if sensor_resolution not in Mcp9600_sensor_resolutions:
      raise ValueError("sensor_resolution should be in {}".format(list(
        Mcp9600_sensor_resolutions.keys())))
    else:
      self._sensor_resolution = sensor_resolution

    if adc_resolution not in Mcp9600_adc_resolutions:
      raise ValueError("adc_resolution should be in {}".format(list(
        Mcp9600_adc_resolutions.keys())))
    else:
      self._adc_resolution = adc_resolution

    if thermocouple_type not in Mcp9600_thermocouple_types:
      raise ValueError("thermocouple_type should be in {}".format(list(
        Mcp9600_thermocouple_types.keys())))
    else:
      self._thermocouple_type = thermocouple_type

    if filter_coefficient not in Mcp9600_filter_coefficients:
      raise ValueError("filter_coefficient should be an integer between "
                       "0 and 7")
    else:
      self._filter_coefficient = filter_coefficient

    if mode not in Mcp9600_modes:
      raise ValueError("mode should be in {}".format(Mcp9600_modes))
    else:
      self._mode = mode

  def open(self) -> None:
    """Initializes the I2C communication and the device."""

    if not self._is_connected():
      raise IOError("The MCP9600 is not connected")
    self.log(logging.INFO, "Setting up the MCP9600")

    # Setting the sensor according to the user parameters
    config_sensor = Mcp9600_filter_coefficients[self._filter_coefficient]
    config_sensor |= Mcp9600_thermocouple_types[self._thermocouple_type]

    self.log(logging.DEBUG,
             f"Writing {config_sensor} to the register "
             f"{Mcp9600_registers['Thermocouple Sensor Configuration']} "
             f"at address {self._device_address}")
    self._bus.write_i2c_block_data(self._device_address, Mcp9600_registers[
        'Thermocouple Sensor Configuration'], [config_sensor])

    # Setting the device according to the user parameters
    config_device = Mcp9600_sensor_resolutions[self._sensor_resolution]
    config_device |= Mcp9600_adc_resolutions[self._adc_resolution]
    config_device |= 0b00  # Normal operating mode
    self.log(logging.DEBUG,
             f"Writing {config_device} to the register "
             f"{Mcp9600_registers['Device Configuration']} "
             f"at address {self._device_address}")
    self._bus.write_i2c_block_data(self._device_address,
                                   Mcp9600_registers['Device Configuration'],
                                   [config_device])

  def get_data(self) -> list[float]:
    """Reads the registers containing the conversion result.

    The output is in `°C` for all modes except the raw data ADC one, which
    outputs Volts.

    Returns:
      A :obj:`list`: containing the timestamp and the output value.
    """

    # Starting a conversion
    value = self._bus.read_i2c_block_data(self._device_address,
                                          Mcp9600_registers['Status'], 1)[0]
    self.log(logging.DEBUG, f"Read {value} from register "
                            f"{Mcp9600_registers['Status']} at address "
                            f"{self._device_address}")
    self.log(logging.DEBUG,
             f"Writing {value & 0xBF} to the register "
             f"{Mcp9600_registers['Status']} at address "
             f"{self._device_address}")
    self._bus.write_i2c_block_data(self._device_address,
                                   Mcp9600_registers['Status'],
                                   [value & 0xBF])

    # Waiting for the conversion to complete
    t0 = time()
    while not self._data_available():
      if time() - t0 > 1:
        raise TimeoutError('Waited too long for data to be ready !')

    out = [time()]

    # The number of output bits varies according to the selected mode
    # Temperature-modes outputs are 2 bytes long
    if self._mode in ['Hot Junction Temperature',
                      'Junction Temperature Delta',
                      'Cold Junction Temperature']:
      block = self._bus.read_i2c_block_data(self._device_address,
                                            Mcp9600_registers[self._mode], 2)
      self.log(logging.DEBUG, f"Read {block} from register "
                              f"{Mcp9600_registers[self._mode]} at address "
                              f"{self._device_address}")
      value_raw = ((block[0] << 8) | block[1])

      # Converting the raw output value into °C
      value = value_raw / 16
      if block[0] >> 7:
        value -= 4096
      out.append(value)

    # Raw ADC output is 3 bytes long
    else:
      block = self._bus.read_i2c_block_data(self._device_address,
                                            Mcp9600_registers[self._mode], 3)
      self.log(logging.DEBUG, f"Read {block} from register "
                              f"{Mcp9600_registers[self._mode]} at address "
                              f"{self._device_address}")
      value_raw = (block[0] << 16) | (block[1] << 8) | block[2]

      # Converting the raw output value into Volts
      # The 2µV resolution is given in the MCP9600 datasheet
      if block[0] >> 1 & 1:
        value_raw -= 2 ** 18

      out.append(value_raw * 2E-6)

    return out

  def close(self) -> None:
    """Switches the MCP9600 to shut down mode and closes the I2C bus."""

    if self._bus is not None:
      # Switching to shut down mode, keeping configuration
      value = self._bus.read_i2c_block_data(self._device_address,
                                            Mcp9600_registers[
                                             'Device Configuration'], 1)[0]
      self.log(logging.DEBUG, f"Read {value} from register "
                              f"{Mcp9600_registers['Device Configuration']}"
                              f" at address {self._device_address}")
      value &= 0xFD
      value |= 0x01
      self.log(logging.DEBUG,
               f"Writing {value} to the register "
               f"{Mcp9600_registers['Device Configuration']} at address "
               f"{self._device_address}")
      self._bus.write_i2c_block_data(self._device_address,
                                     Mcp9600_registers['Device Configuration'],
                                     [value])
      self.log(logging.INFO, "Closing the I2C connection to the MCP9600")
      self._bus.close()

  def _data_available(self) -> int:
    """Reads the data available bit.

    Returns: `1` if data is available, else `0`
    """

    status = self._bus.read_i2c_block_data(self._device_address,
                                           Mcp9600_registers['Status'], 1)[0]
    self.log(logging.DEBUG,
             f"Read {status} from register {Mcp9600_registers['Status']}"
             f" at address {self._device_address}")

    # The MCP9600 features an over-temperature protection
    if status & 0x10:
      raise ValueError("Too hot for selected thermocouple !")

    return status & 0x40

  def _is_connected(self) -> bool:
    """Tries reading a byte from the device.

    Returns: :obj:`True` if reading was successful, else :obj:`False`
    """

    try:
      self._bus.read_byte(self._device_address)
      return True
    except IOError:
      return False
