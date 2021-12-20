# coding: utf-8

from time import time
from typing import Union
from .inout import InOut
from .._global import OptionalModule
from ..tool import ft232h_server as ft232h, Usb_server, i2c_msg_ft232h

try:
  import adafruit_mprls
except (ImportError, ModuleNotFoundError):
  adafruit_mprls = OptionalModule('adafruit_mprls')

try:
  import board
except (ImportError, ModuleNotFoundError):
  board = OptionalModule('board', 'Blinka is necessary to use the I2C bus')

try:
  from digitalio import DigitalInOut
except (ImportError, ModuleNotFoundError):
  DigitalInOut = OptionalModule('digitalio',
                                'Blinka is necessary to access the GPIOs')

try:
  from smbus2 import SMBus, i2c_msg
except (ImportError, ModuleNotFoundError):
  SMBus = i2c_msg = OptionalModule('smbus2')

try:
  import RPi.GPIO as GPIO
except (ModuleNotFoundError, ImportError):
  GPIO = OptionalModule("RPi.GPIO")

mprls_status_bits = {'busy': 0x20,
                     'memory error': 0x04,
                     'math saturation': 0x01}
mprls_backends = ['Pi4', 'blinka', 'ft232h']


class Mprls(Usb_server, InOut):
  """The Mprls inout is meant for reading pressure from Adafruit's Mprls
    pressure sensor.

    It communicates over I2C with the sensor.
    """

  def __init__(self,
               backend: str,
               eoc_pin: Union[str, int] = None,
               device_address: int = 0x18,
               i2c_port: int = 1,
               ft232h_ser_num: str = None) -> None:
    """Initializes the parent class and opens the I2C bus.

    Args:
      backend (:obj:`str`): Should be one of :
        ::

          'Pi4', 'blinka', 'ft232h'

        The `'Pi4'` backend is optimized but only works on boards supporting
        the :mod:`smbus2` module, like the Raspberry Pis. The `'blinka'`
        backend may be less performant and requires installing Adafruit's
        modules, but these modules are compatible with and maintained on a wide
        variety of boards. The `'ft232h'` backend allows controlling the
        MPRLS from a PC using Adafruit's FT232H USB to I2C adapter. See
        :ref:`Crappy for embedded hardware` for details.
      eoc_pin (:obj:`int` or :obj:`str`, optional): Optionally, reads the end
        of conversion signal from a GPIO rather than from an I2C message.
        Speeds up the reading and decreases the traffic on the bus, but
        requires one extra wire. With the backend `'Pi4'`, give the index of
        the GPIO in BCM convention. With the `'ft232h'` backend, give the name
        of the GPIO in the format `Dx` or `Cx`. With the backend `'blinka'`,
        it should be a string but the syntax varies according to the board.
        Refer to blinka's documentation for more information.
      device_address (:obj:`int`, optional): The I2C address of the MPRLS.
        The address of the devices sold by Adafruit is `0x18`, but other
        suppliers may sell it with another address.
      i2c_port (:obj:`int`, optional): The I2C port over which the MPRLS
        should communicate. On most Raspberry Pi models the default I2C port is
        `1`.
      ft232h_ser_num (:obj:`str`, optional): If backend is `'ft232h'`, the
        serial number of the FT232H to use for communication.
    """

    if not isinstance(backend, str) or backend not in mprls_backends:
      raise ValueError("backend should be in {}".format(mprls_backends))
    self._backend = backend

    Usb_server.__init__(self,
                        serial_nr=ft232h_ser_num if ft232h_ser_num else '',
                        backend=backend)
    queue, block_number, namespace, command_event, \
        answer_event, next_event, done_event = super().start_server()

    InOut.__init__(self)

    if backend == 'ft232h':
      self._bus = ft232h(mode='I2C',
                         queue=queue,
                         namespace=namespace,
                         command_event=command_event,
                         answer_event=answer_event,
                         block_number=block_number,
                         next_block=next_event,
                         done_event=done_event,
                         serial_nr=ft232h_ser_num)

    if not isinstance(device_address, int):
      raise TypeError("device_address should be an integer.")
    self._address = device_address

    if not isinstance(i2c_port, int):
      raise TypeError("i2c_port should be an integer.")
    self._i2c_port = i2c_port

    if eoc_pin is not None:
      if backend == 'blinka' and not isinstance(eoc_pin, str):
        raise TypeError('eoc_pin should be a string when using the blinka '
                        'backend !')
      elif backend == 'ft232h' and not isinstance(eoc_pin, str):
        raise TypeError('eoc_pin should be a string when using the ft232h '
                        'backend !')
      elif backend == 'Pi4' and not isinstance(eoc_pin, int):
        raise TypeError('eoc_pin should be an int when using the Pi4 '
                        'backend !')
    self._eoc_pin = eoc_pin

  def open(self) -> None:
    """Opens the I2C bus."""

    if self._backend == 'Pi4':
      self._bus = SMBus(self._i2c_port)
      if self._eoc_pin is not None:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._eoc_pin, GPIO.IN)

    elif self._backend == 'blinka':
      if self._eoc_pin is not None:
        eoc = DigitalInOut(getattr(board, self._eoc_pin))
      else:
        eoc = None
      self._mpr = adafruit_mprls.MPRLS(board.I2C(),
                                       psi_min=0,
                                       psi_max=25,
                                       eoc_pin=eoc)

    self._i2c_msg = i2c_msg_ft232h if self._backend == 'ft232h' else i2c_msg

  def get_data(self) -> list:
    """Reads the pressure value.

    Returns:
      The timestamp and the pressure value in hPa.
    """

    if self._backend == 'blinka':
      pres = self._mpr.pressure

    else:
      # Starting conversion
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

  def close(self):
    """Closes the I2C bus."""

    if self._backend != 'blinka':
      self._bus.close()

    if self._backend == 'Pi4' and self._eoc_pin is not None:
      GPIO.cleanup()

  def _data_available(self) -> bool:
    """Returns :obj:`True` if data is available, :obj:`False` otherwise."""

    # EOC signal from the I2C communication
    if self._eoc_pin is None:
      wait = self._i2c_msg.read(self._address, 1)
      self._bus.i2c_rdwr(wait)
      return not list(wait)[0] & mprls_status_bits['busy']
    # EOC signal from a GPIO
    elif self._backend == 'ft232h':
      return bool(self._bus.get_gpio(self._eoc_pin))
    else:
      return bool(GPIO.input(self._eoc_pin))
