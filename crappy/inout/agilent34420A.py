# coding: utf-8

from time import time
from .inout import InOut
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


class Agilent34420a(InOut):
  """Sensor class for Agilent34420A devices.

  This class contains method to measure values of resistance or voltage on
  Agilent34420A devices.

  Note:
    May work for other devices too, but not tested.

    If you have issues with this class returning a lot of `'bad serial'`, make
    sure you have the last version of :mod:`serial`.
  """

  def __init__(self,
               mode: bytes = b"VOLT",
               device: str = '/dev/ttyUSB0',
               baudrate: int = 9600,
               timeout: float = 1) -> None:
    """Sets the args and initializes parent class.

    Args:
      mode (:obj:`bytes`, optional): Desired value to measure. Should be either
        `b'VOLT'` or `b'RES'`.
      device (:obj:`str`, optional): Path to the device.
      baudrate (:obj:`int`, optional): Desired baudrate.
      timeout (:obj:`float`, optional): Timeout for the serial connection.
    """

    InOut.__init__(self)
    # path to the device
    self.device = device
    # desired baudrate
    self.baudrate = baudrate
    # timeout for the serial connection
    self.timeout = timeout
    # desired value to measure
    self.mode = mode

  def open(self) -> None:
    self.ser = serial.Serial(port=self.device, baudrate=self.baudrate,
                             timeout=self.timeout)
    self.ser.write(b"*RST;*CLS;*OPC?\n")
    self.ser.write(b"SENS:FUNC \"" + self.mode + b"\";  \n")
    self.ser.write(b"SENS:" + self.mode + b":NPLC 2  \n")
    # ser.readline()
    self.ser.write(b"SYST:REM\n")
    self.get_data()

  def get_data(self) -> list:
    """Reads the signal, returns :obj:`False` if error and prints
    `'bad serial'`."""

    self.ser.write(b"READ?  \n")
    t = time()
    try:
      return [t, float(self.ser.readline())]
    except (serial.SerialException, ValueError):
      self.ser.flush()
      return [t, 0]

  def close(self) -> None:
    """Closes the serial port."""

    self.ser.close()
