# coding: utf-8

from time import time
from re import findall
from typing import Union
from .inout import InOut
from ..tool import ft232h_server as ft232h, Usb_server
from .._global import OptionalModule

try:
  from smbus2 import SMBus
except (ModuleNotFoundError, ImportError):
  smbus2 = OptionalModule("smbus2")

try:
  import RPi.GPIO as GPIO
except (ModuleNotFoundError, ImportError):
  GPIO = OptionalModule("RPi.GPIO")

try:
  import board
except (ModuleNotFoundError, ImportError):
  board = OptionalModule('board', 'Blinka is necessary to use the I2C bus')

try:
  import busio
except (ModuleNotFoundError, ImportError):
  busio = OptionalModule('busio', 'Blinka is necessary to use the I2C bus')

try:
  import adafruit_ads1x15.ads1115 as ads
  from adafruit_ads1x15.analog_in import AnalogIn
except (ModuleNotFoundError, ImportError):
  ADS = OptionalModule('adafruit_ads1x15',
                       'Blinka is necessary to use the I2C bus')
  AnalogIn = OptionalModule('AnalogIn',
                            'Blinka is necessary to use the I2C bus')

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

Ads1115_blinka_gain = {0.256: 16,
                       0.512: 8,
                       1.024: 4,
                       2.048: 2,
                       4.096: 1,
                       6.144: 2/3}

Ads1115_backends = ['Pi4', 'ft232h', 'blinka']


class Ads1115(Usb_server, InOut):
  """A class for controlling Adafruit's ADS1115 16-bits ADC.

  The Ads1115 InOut block is meant for reading output values from a 16-bits
  ADS1115 ADC, using the I2C protocol. The output is in Volts by default, but a
  ``gain`` and an ``offset`` can be specified.
  """

  def __init__(self,
               backend: str,
               device_address: int = 0x48,
               i2c_port: int = 1,
               sample_rate: int = 128,
               v_range: float = 2.048,
               multiplexer: str = 'A1',
               dry_pin: Union[str, int] = None,
               gain: float = 1,
               offset: float = 0,
               ft232h_ser_num: str = None) -> None:
    """Checks arguments validity.

    Args:
      backend (:obj:`str`): The backend for communicating with the ADS1115.
        Should be one of:
        ::

          'Pi4', 'ft232h', 'blinka'

        The `'Pi4'` backend is optimized but only works on boards supporting
        the :mod:`smbus2` module, like the Raspberry Pis. The `'blinka'`
        backend may be less performant and requires installing Adafruit's
        modules, but these modules are compatible with and maintained on a wide
        variety of boards. The `'ft232h'` backend allows controlling the
        ADS1115 from a PC using Adafruit's FT232H USB to I2C adapter. See
        :ref:`Crappy for embedded hardware` for details.
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
        signal corresponding to the `0x7FFF` output in bits, i.e. that
        saturates the sensor. A signal of ``-v_range`` Volts gives a `0x8000`
        output in bits. Available ``v_range`` values are:
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

      dry_pin (:obj:`int` or :obj:`str`, optional): Optionally, reads the end
        of conversion signal from a GPIO rather than from an I2C message.
        Speeds up the reading and decreases the traffic on the bus, but
        requires one extra wire. With the backend `'Pi4'`, give the index of
        the GPIO in BCM convention. With the `'ft232h'` backend, give the name
        of the GPIO in the format `Dx` or `Cx`. This feature is not available
        with the `'blinka'` backend.
      gain (:obj:`float`, optional): Allows to tune the output value according
        to the formula:
        ::

          output = gain * tension + offset.

      offset (:obj:`float`, optional): Allows to tune the output value
        according to the formula:
        ::

          output = gain * tension + offset.

      ft232h_ser_num (:obj:`str`, optional): If backend is `'ft232h'`, the
        serial number of the FT232H to use for communication.

    Warning:
      AINx voltages should not be higher than `VDD+0.3V` nor lower than
      `GND-0.3V`. Setting high ``v_range`` values does not allow measuring
      voltages higher than `VDD` !!
    """

    if not isinstance(backend, str) or backend not in Ads1115_backends:
      raise ValueError("backend should be in {}".format(Ads1115_backends))
    self._backend = backend

    Usb_server.__init__(self,
                        serial_nr=ft232h_ser_num if ft232h_ser_num else '',
                        backend=backend)
    InOut.__init__(self)
    queue, block_number, namespace, command_event, \
        answer_event, next_event, done_event = super().start_server()

    if backend == 'Pi4':
      self._bus = SMBus(i2c_port)
    elif backend == 'ft232h':
      self._bus = ft232h(mode='I2C',
                         queue=queue,
                         namespace=namespace,
                         command_event=command_event,
                         answer_event=answer_event,
                         block_number=block_number,
                         next_block=next_event,
                         done_event=done_event,
                         serial_nr=ft232h_ser_num)
    elif backend == 'blinka':
      i2c = busio.I2C(board.SCL, board.SDA)
      self._ads = ads.ADS1115(i2c)

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

    if dry_pin is not None:
      if backend == 'ft232h' and not isinstance(dry_pin, str):
        raise TypeError('int_pin should be a string when using the ft232h '
                        'backend !')
      elif backend == 'Pi4' and not isinstance(dry_pin, int):
        raise TypeError('int_pin should be an int when using the Pi4 '
                        'backend !')
      elif backend == 'blinka' and dry_pin is not None:
        print("Warning ! Getting the end of conversion information from the"
              "DRY pin is not supported by the blinka backend.")
    self._dry_pin = dry_pin

    self._gain = gain
    self._offset = offset

  def open(self) -> None:
    """Sets the I2C communication and device."""

    if self._backend == 'blinka':
      self._ads.gain = Ads1115_blinka_gain[self._v_range]
      self._ads.data_rate = self._sample_rate
      chan_nr = findall(r'\d', self._multiplexer)
      if len(chan_nr) == 1:
        self._chan = AnalogIn(self._ads,
                              getattr(ads, 'P{}'.format(chan_nr[0])))
      else:
        self._chan = AnalogIn(self._ads,
                              getattr(ads, 'P{}'.format(chan_nr[0])),
                              getattr(ads, 'P{}'.format(chan_nr[1])))

    else:
      if not self._is_connected():
        raise IOError("The ADS1115 is not connected")

      # Setting the configuration register according to the user values
      init_value = Ads1115_config_mux[self._multiplexer]
      init_value |= Ads1115_config_gain[self._v_range]
      init_value |= 0x0100  # Single shot operating mode
      init_value |= Ads1115_config_dr[self._sample_rate]
      if self._dry_pin is None:
        # Setting the two last bits to 11 to disable the DRY pin
        init_value |= 0x0003

      self._set_register(Ads1115_pointer_config, init_value)

      # Setting the threshold registers to activate the DRY pin output
      if self._dry_pin is not None:
        self._set_register(Ads1115_pointer_lo_thresh, 0x0000)
        self._set_register(Ads1115_pointer_hi_thresh, 0xFFFF)

      if self._backend == 'Pi4' and self._dry_pin is not None:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._dry_pin, GPIO.IN)

  def get_data(self) -> list:
    """Reads the registers containing the conversion result.

    The output is in Volts, unless a gain and offset are applied.

    Returns:
      :obj:`list`: A list containing the timeframe and the voltage value
    """

    if self._backend == 'blinka':
      out = [time()]
      value = self._chan.voltage

    else:
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

      # Converting the output value into Volts
      value_raw = (ms_byte << 8) | ls_byte
      if ms_byte >> 7:
        value_raw -= 2 ** 16
      value = value_raw * self._v_range / 2 ** 15

    out.append(self._offset + self._gain * value)
    return out

  def close(self) -> None:
    """Closes the I2C bus"""

    if self._backend != 'blinka':
      self._bus.close()

    if self._backend == 'Pi4' and self._dry_pin is not None:
      GPIO.cleanup()

  def _set_register(self, register_address: int, value: int) -> None:
    """Thin wrapper for writing data to the registers."""

    self._bus.write_i2c_block_data(self._device_address, register_address,
                                   [(value >> 8) & 0xFF, value & 0xFF])

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

  def _data_available(self) -> bool:
    """Returns :obj:`True` if data is available, :obj:`False` otherwise."""

    # EOC signal from the I2C communication
    if self._dry_pin is None:
      return self._bus.read_i2c_block_data(self._device_address,
                                           Ads1115_pointer_config,
                                           1)[0] & 0x80
    # EOC signal from a GPIO
    elif self._backend == 'ft232h':
      return not bool(self._bus.get_gpio(self._dry_pin))
    else:
      return not bool(GPIO.input(self._dry_pin))
