# coding: utf-8

from time import time
from typing import List
import logging

from .meta_inout import InOut
from .._global import OptionalModule

try:
  from smbus2 import SMBus
except (ModuleNotFoundError, ImportError):
  SMBus = OptionalModule("smbus2")

try:
  import board
except (ModuleNotFoundError, ImportError):
  board = OptionalModule('board', 'Blinka is necessary to use the I2C bus')

try:
  import busio
except (ModuleNotFoundError, ImportError):
  busio = OptionalModule('busio', 'Blinka is necessary to use the I2C bus')

try:
  from adafruit_bus_device.i2c_device import I2CDevice
except (ModuleNotFoundError, ImportError):
  I2CDevice = OptionalModule('adafruit_bus_device',
                             'Blinka is necessary to use the I2C bus')

try:
  from adafruit_mcp9600 import MCP9600 as MCP9600Adafruit
except (ModuleNotFoundError, ImportError):
  MCP9600Adafruit = OptionalModule('adafruit_mcp9600',
                                   'Blinka is necessary to use the I2C bus')

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

Mcp9600_backends = ['Pi4', 'blinka']


class MCP9600(InOut):
  """This class can read temperature values from an MCP9600 thermocouple
  reader.

  It communicates over the I2C protocol. The output is in `°C`, except for one
  operating mode that returns Volts. Several parameters can be tuned, like the
  thermocouple type, the reading resolution or the filter coefficient. Note
  that the MCP9600 can only achieve a data rate of a few Hz.
  """

  def __init__(self,
               backend: str,
               thermocouple_type: str,
               i2c_port: int = 1,
               device_address: int = 0x67,
               adc_resolution: int = 18,
               sensor_resolution: float = 0.0625,
               filter_coefficient: int = 0,
               mode: str = 'Hot Junction Temperature') -> None:
    """Checks the validity of the arguments.

    Args:
      backend: The backend for communicating with the MCP9600.
        Should be one of:
        ::

          'Pi4', 'blinka'

        The `'Pi4'` backend is optimized but only works on boards supporting
        the :mod:`smbus2` module, like the Raspberry Pis. The `'blinka'`
        backend may be less performant and requires installing
        :mod:`Adafruit-Blinka`, :mod:`adafruit-circuitpython-busdevice` and
        :mod:`adafruit-circuitpython-mcp9600`, but these modules are compatible
        with and maintained on a wide variety of boards.
      thermocouple_type: The type of thermocouple connected to the MCP9600. The
        possible types are:
        ::

          'J', 'K', 'T', 'N', 'S', 'E', 'B', 'R'

      i2c_port: The I2C port over which the MCP9600 should communicate. On most
        Raspberry Pi models the default I2C port is `1`.
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

    """

    self._bus = None
    self._mcp = None

    if not isinstance(backend, str) or backend not in Mcp9600_backends:
      raise ValueError("backend should be in {}".format(Mcp9600_backends))
    self._backend = backend

    super().__init__()

    if backend == 'Pi4':
      self._bus = SMBus(i2c_port)
    elif backend == 'blinka':
      self._i2c = busio.I2C(board.SCL, board.SDA, frequency=200000)

    self._device_address = device_address

    if sensor_resolution not in Mcp9600_sensor_resolutions:
      raise ValueError("sensor_resolution should be in {}".format(list(
        Mcp9600_sensor_resolutions.keys())))
    else:
      self._sensor_resolution = sensor_resolution

    if adc_resolution not in Mcp9600_adc_resolutions:
      raise ValueError("adc_resolution should be in {}".format(list(
        Mcp9600_adc_resolutions.keys())))
    elif adc_resolution != 18 and backend == 'blinka':
      raise ValueError("It is not possible to set adc_resolution using the"
                       "backend blinka.")
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
    elif mode == 'Raw Data ADC' and backend == 'blinka':
      raise ValueError('The Raw Data ADC mode is not available using the '
                       'backend blinka.')
    else:
      self._mode = mode

  def open(self) -> None:
    """Initializes the I2C communication and the device."""

    if self._backend == 'blinka':
      self.log(logging.INFO, "Connecting the the MCP9600 with backend blinka")
      self._mcp = MCP9600Adafruit(self._i2c,
                                  address=self._device_address,
                                  tctype=self._thermocouple_type,
                                  tcfilter=self._filter_coefficient)
      self._mcp.ambient_resolution = self._sensor_resolution == 0.25

    else:
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

  def get_data(self) -> List[float]:
    """Reads the registers containing the conversion result.

    The output is in `°C` for all modes except the raw data ADC one, which
    outputs Volts.

    Returns:
      A :obj:`list`: containing the timestamp and the output value.
    """

    if self._backend == 'blinka':
      out = [time()]
      if self._mode == 'Hot Junction Temperature':
        out.append(self._mcp.temperature)
      elif self._mode == 'Junction Temperature Delta':
        out.append(self._mcp.delta_temperature)
      elif self._mode == 'Cold Junction Temperature':
        out.append(self._mcp.ambient_temperature)

    else:
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
    """Switches the MCP9600 to shutdown mode and closes the I2C bus."""

    if self._backend != 'blinka' and self._bus is not None:
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
