# coding: utf-8

import time
from .inout import InOut
from ..tool import ft232h
from .._global import OptionalModule

try:
  import smbus2
except (ModuleNotFoundError, ImportError):
  smbus2 = OptionalModule("smbus2")

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

Mcp9600_backends = ['Pi4', 'ft232h']


class Mcp9600(InOut):
  """Class for controlling Adafruit's MCP9600 thermocouple reader.

  The Mcp9600 InOut block is meant for reading temperature from an MCP9600
  board, using the I2C protocol. The output is in `°C`, except for one
  operating mode that returns Volts.

  Warning:
    Only available on Raspberry Pi for now !
  """

  def __init__(self,
               backend: str,
               thermocouple_type: str,
               i2c_port: int = 1,
               device_address: int = 0x67,
               adc_resolution: int = 18,
               sensor_resolution: float = 0.0625,
               filter_coefficient: int = 0,
               mode: str = 'Hot Junction Temperature',
               ft232h_ser_num: str = None) -> None:
    """Checks arguments validity.

    Args:
      backend (:obj:`str`): The backend for communicating with the NAU7802.
        Should be one of:
        ::

          'Pi4', 'ft232h'

      thermocouple_type (:obj:`str`): The type of thermocouple plugged in the
        MCP9600. The available types are:
        ::

          'J', 'K', 'T', 'N', 'S', 'E', 'B', 'R'

      i2c_port(:obj:`int`, optional): The I2C port over which the MCP9600
        should communicate. On most Raspberry Pi models the default I2C port is
        `1`.
      device_address(:obj:`int`, optional): The I2C address of the MCP9600. The
        default address is `0x67`, but it is possible to change this setting
        using a specific setup involving the `ADDR` pin.
      adc_resolution(:obj:`int`, optional): The number of bits the ADC output
        is encoded on. The greater the resolution, the lower the sample rate.
        The available resolutions are:
        ::

          12, 14, 16, 18

      sensor_resolution(:obj:`float`, optional): The temperature measurement
        resolution in `°C`. It should be either `0.0625` or `0.25`. Setting the
        resolution to `0.25` will increase the sample rate, but the output
        temperature will be encoded on two bits less.
      filter_coefficient(:obj:`int`, optional): The MCP9600 features an
        integrated filter (see its documentation for the exact filter formula).
        When set to `0`, the filter is inactive. It is maximal when set to `7`.
        When active, the filter will prohibit fast temperature changes, thus
        limiting noise and smoothening the signal.
      mode(:obj:`str`, optional): Four different values can be accessed when
        measuring a temperature: the temperature of the thermocouple (hot
        junction temperature), the temperature of the MCP9600 board (cold
        junction temperature), the temperature calculated from the ADC data and
        thermocouple type but not yet cold junction-compensated (junction
        temperature delta), and the raw ADC measurement of the voltage
        difference in the thermocouple (raw data ADC, in Volts). The available
        modes are thus:
        ::

          'Hot Junction Temperature',
          'Junction Temperature Delta',
          'Cold Junction Temperature',
          'Raw Data ADC'

      ft232h_ser_num (:obj:`str`, optional): If backend is `'ft232h'`, the
        serial number of the ft232h to use for communication.

    """

    InOut.__init__(self)
    if backend not in Mcp9600_backends:
      raise ValueError("backend should be in {}".format(Mcp9600_backends))

    if backend == 'Pi4':
      self._bus = smbus2.SMBus(i2c_port)
    else:
      self._bus = ft232h('I2C', ft232h_ser_num)
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
    """Sets the I2C communication and device."""

    if not self._is_connected():
      raise IOError("The MCP9600 is not connected")

    # Setting the sensor according to the user parameters
    config_sensor = 0x00
    config_sensor |= Mcp9600_filter_coefficients[self._filter_coefficient]
    config_sensor |= Mcp9600_thermocouple_types[self._thermocouple_type]

    self._bus.write_i2c_block_data(self._device_address, Mcp9600_registers[
        'Thermocouple Sensor Configuration'], [config_sensor])

    # Setting the device according to the user parameters
    config_device = 0x00
    config_device |= Mcp9600_sensor_resolutions[self._sensor_resolution]
    config_device |= Mcp9600_adc_resolutions[self._adc_resolution]
    config_device |= 0b00  # Normal operating mode
    self._bus.write_i2c_block_data(self._device_address,
                                   Mcp9600_registers['Device Configuration'],
                                   [config_device])

  def get_data(self) -> list:
    """Reads the registers containing the conversion result.

    The output is in `°C` for all modes except the raw data ADC one, which
    outputs Volts.

    Returns:
      :obj:`list`: A list containing the timeframe and the output value
    """

    # Starting a conversion
    value = self._bus.read_i2c_block_data(self._device_address,
                                          Mcp9600_registers['Status'], 1)[0]
    value &= 0xBF
    self._bus.write_i2c_block_data(self._device_address,
                                   Mcp9600_registers['Status'], [value])

    # Waiting for the conversion to complete
    t_wait = 0
    while not self._data_available():
      time.sleep(0.01)
      t_wait += 0.01
      if t_wait > 1:
        raise TimeoutError

    # The MCP9600 features an over-temperature protection
    if self._bus.read_i2c_block_data(self._device_address,
                                     Mcp9600_registers['Status'], 1)[0] \
            >> 4 & 1:
      raise ValueError("Too hot for selected thermocouple !")

    out = [time.time()]

    # The number of output bits varies according to the selected mode
    # Temperature-modes outputs are 2 bytes long
    if self._mode in ['Hot Junction Temperature', 'Junction Temperature Delta',
                      'Cold Junction Temperature']:
      block = self._bus.read_i2c_block_data(self._device_address,
                                            Mcp9600_registers[self._mode], 2)
      value_raw = (block[0] << 8) | block[1]

      # Converting the raw output value into °C
      if self._mode == 'Cold Junction Temperature':
        if block[0] >> 4 & 1:
          out.append((-(2 ** 11 - 1) + (value_raw & 0x0FFF)) / 16)
        else:
          out.append(value_raw / 16)
      else:
        if block[0] >> 7:
          out.append((-(2 ** 15 - 1) + (value_raw & 0x7FFF)) / 16)
        else:
          out.append(value_raw / 16)

    # Raw ADC output is 3 bytes long
    else:
      block = self._bus.read_i2c_block_data(self._device_address,
                                            Mcp9600_registers[self._mode], 3)
      value_raw = (block[0] << 16) | (block[1] << 8) | block[2]
      shift = 18 - self._adc_resolution

      # Converting the raw output value into Volts
      # The 0.262 factor is given in the MCP9600 datasheet
      if block[0] >> 1 & 1:
        value = (-(2 ** (self._adc_resolution - 1) - 1) + (
                  (value_raw & 0x01FFFF) >> shift)) / (
                          2 ** (self._adc_resolution - 1) - 1)
        out.append(value * 2 ** 18 / 10 ** 6)
      else:
        value = (value_raw >> shift) / (2 ** (self._adc_resolution - 1) - 1)
        out.append(value * 2 ** 18 / 10 ** 6)

    return out

  def close(self) -> None:
    """Switches the MCP9600 to shutdown mode."""

    # switching to shutdown mode, keeping configuration
    value = self._bus.read_i2c_block_data(self._device_address,
                                          Mcp9600_registers[
                                           'Device Configuration'], 1)[0]
    value &= 0xFD
    value |= 0x01
    self._bus.write_i2c_block_data(self._device_address,
                                   Mcp9600_registers['Device Configuration'],
                                   [value])
    self._bus.close()

  def _data_available(self) -> int:
    """Reads the data available bit.

    Returns: `1` if data is available, else `0`
    """

    return self._bus.read_i2c_block_data(self._device_address,
                                         Mcp9600_registers['Status'], 1)[0] \
        >> 6 & 1

  def _is_connected(self) -> bool:
    """Tries reading a byte from the device.

    Returns: :obj:`True` if reading was successful, else :obj:`False`
    """

    try:
      self._bus.read_byte(self._device_address)
      return True
    except IOError:
      return False
