# coding: utf-8

from time import time
from .inout import InOut
from .._global import OptionalModule

try:
  from pijuice import PiJuice
except (ModuleNotFoundError, ImportError):
  PiJuice = OptionalModule("pijuice")

try:
  from smbus2 import SMBus
except (ImportError, ModuleNotFoundError):
  SMBus = OptionalModule('smbus2')

pijuice_commands = {'status': 0x40,
                    'fault_event': 0x44,
                    'charge_level': 0x41,
                    'button_event': 0x45,
                    'battery_temperature': 0x47,
                    'battery_voltage': 0x49,
                    'battery_current': 0x4B,
                    'io_voltage': 0x4D,
                    'io_current': 0x4F,
                    'led_state': 0x66,
                    'led_blink': 0x68,
                    'io_pin_access': 0x75}

pijuice_battery_status = {0x00: 'Normal',
                          0x04: 'Charging_from_usb',
                          0x08: 'Charging_from_gpio',
                          0x0C: 'Absent'}

pijuice_power_status = {0x00: 'Not_present',
                        0x10: 'Bad',
                        0x20: 'Weak',
                        0x30: 'Present'}

pijuice_backends = ['Pi4', 'pijuice']


class Pijuice(InOut):
  """Block getting various information about a piJuice power platform,
  including the charge level and the power supply status.

  Warning:
    Only available on Raspberry Pi !
  """

  def __init__(self,
               i2c_port: int = 1,
               address: int = 0x14,
               backend: str = 'Pi4') -> None:
    """Checks arguments validity.

    Args:
      backend (:obj:`str`, optional): Should be one of :
        ::

          'Pi4', 'pijuice'

        The `'Pi4'` backend is based on the :mod:`smbus2` module, while the
        `'pijuice'` backend is based on the :mod:`pijuice`.
      i2c_port(:obj:`int`, optional): The I2C port over which the PiJuice
        should communicate.
      address(:obj:`int`, optional): The I2C address of the piJuice. The
        default address is 0x14.
    """

    super().__init__()
    if not isinstance(i2c_port, int):
      raise TypeError("i2c_port should be an int")
    self._i2c_port = i2c_port

    if not isinstance(address, int):
      raise TypeError("address should be an int")
    else:
      self._address = address

    if not isinstance(backend, str) or backend not in pijuice_backends:
      raise ValueError("backend should be in {}".format(pijuice_backends))
    self._backend = backend

  def open(self) -> None:
    """Opens the I2C port."""

    if self._backend == 'Pi4':
      self._bus = SMBus(self._i2c_port)
    elif self._backend == 'pijuice':
      self._pijuice = PiJuice(self._i2c_port, self._address)

  def get_data(self) -> dict:
    """Reads all the available information on the battery status.

    Returns:
      :obj:`dict`: Returns a dict containing:

        - the timeframe in seconds as a :obj:`float` in label ``t(s)``
        - the battery status as a :obj:`str` in label ``battery_status``
        - the USB status as a :obj:`str` in label ``USB_status``
        - the GPIO status as a :obj:`str` in label ``GPIO_status``
        - the charge level as an :obj:`int` in label ``charge_level``
        - the battery temperature in °C as an :obj:`int` in label
          ``battery_temperature``
        - the battery voltage in mV as an :obj:`int` in label
          ``battery_voltage``
        - the battery current in mA as an :obj:`int` in label
          ``battery_current``
        - the GPIO voltage in mV as an :obj:`int` in label ``GPIO_voltage``
        - the GPIO current in mA as an :obj:`int` in label ``GPIO_current``
    """

    # Gets the battery status
    status = dict()

    if self._backend == 'Pi4':
      data = self._read_i2c(pijuice_commands['status'], 2)
      status['battery'] = pijuice_battery_status[data[0] & 0x0C]
      status['USB'] = pijuice_power_status[data[0] & 0x30]
      status['GPIO'] = pijuice_power_status[data[0] & 0xC0]

    elif self._backend == 'pijuice':
      stat = self._pijuice.status.GetStatus()
      status['battery'] = stat['data']['battery']
      status['USB'] = stat['data']['powerInput']
      status['GPIO'] = stat['data']['powerInput5vIo']

    # Gets the battery charge level
    if self._backend == 'Pi4':
      data = self._read_i2c(pijuice_commands['charge_level'], 2)
      charge_level = data[0]
    else:
      data = self._pijuice.status.GetChargeLevel()
      charge_level = data['data']

    # Gets the battery temperature
    if self._backend == 'Pi4':
      data = self._read_i2c(pijuice_commands['battery_temperature'], 3)
      temp = data[0]
      if temp & 0x80:
        temp -= 0X100
    else:
      data = self._pijuice.status.GetBatteryTemperature()
      temp = data['data']

    # Gets the battery voltage (mV)
    if self._backend == 'Pi4':
      data = self._read_i2c(pijuice_commands['battery_voltage'], 3)
      voltage = (data[1] << 8) | data[0]
    else:
      data = self._pijuice.status.GetBatteryVoltage()
      voltage = data['data']

    # Gets the battery current (mA)
    if self._backend == 'Pi4':
      data = self._read_i2c(pijuice_commands['battery_current'], 3)
      current = (data[1] << 8) | data[0]
      if current & 0x8000:
        current -= 0x10000
    else:
      data = self._pijuice.status.GetBatteryCurrent()
      current = data['data']

    # Gets the GPIO voltage (mV)
    if self._backend == 'Pi4':
      data = self._read_i2c(pijuice_commands['io_voltage'], 3)
      io_voltage = (data[1] << 8) | data[0]
    else:
      data = self._pijuice.status.GetIoVoltage()
      io_voltage = data['data']

    # Gets the GPIO current (mA)
    if self._backend == 'Pi4':
      data = self._read_i2c(pijuice_commands['io_current'], 3)
      io_current = (data[1] << 8) | data[0]
      if io_current & 0x8000:
        io_current -= 0x10000
    else:
      data = self._pijuice.status.GetIoCurrent()
      io_current = data['data']

    return {'t(s)': time(),
            'battery_status': status['battery'],
            'USB_status': status['USB'],
            'GPIO_status': status['GPIO'],
            'charge_level': charge_level,
            'battery_temperature': temp,
            'battery_voltage': voltage,
            'battery_current': current,
            'GPIO_voltage': io_voltage,
            'GPIO_current': io_current}

  def close(self) -> None:
    """Closes the I2C bus."""

    if self._backend == 'Pi4':
      self._bus.close()

  @staticmethod
  def _checksum(data: list) -> list:
    """Compares the received checksum (last byte) to the one calculated over
    the other bytes.

    Also tries with the MSB of the first byte set to 1, as it is apparently
    sometimes 0 when it should be 1.

    Returns:
      The list of data bytes, with the MSB of the first byte corrected if
      needed.

    Raises:
      Raises an :exc:`IOError` if the checksums don't match.
    """

    # First checking on the raw data
    check = 0xFF
    for x in data[:-1]:
      check ^= x

    # If needed, modify the MSB of the first byte and retry
    if check != data[-1]:
      data[0] |= 0x80
      check = 0xFF
      for x in data[:-1]:
        check ^= x

      # If still not successful, raise error
      if check != data[-1]:
        raise IOError("An error occurred, the checksums don't match !")

    return data[:-1]

  def _read_i2c(self, command: int, length: int) -> list:
    """Thin wrapper to reduce verbosity."""

    ret = self._bus.read_i2c_block_data(i2c_addr=self._address,
                                        register=command,
                                        length=length)
    return self._checksum(ret)
