# coding: utf-8

from time import time
from .inout import InOut
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


class Agilent34420a(InOut):
  """
  Sensor class for Agilent34420A devices.

  This class contains method to measure values of resistance or voltage
  on Agilent34420A devices.

  Note:
    May work for other devices too, but not tested.

    If you have issues with this class returning a lot of 'bad serial', \
    make sure you have the last version of pySerial.

  Args:
    - mode ({"VOLT", "RES"}, default: "VOLT"): Desired value to measure.
    - device (str, default: '/dev/ttyUSB0'): Path to the device.
    - baudrate (int, default: 9600): Desired baudrate.
    - timeout (int or float, default: 10): Timeout for the serial connection.

  """

  def __init__(self, mode=b"VOLT", device='/dev/ttyUSB0',
      baudrate=9600, timeout=1):

    InOut.__init__(self)
    # path to the device
    self.device = device
    # desired baudrate
    self.baudrate = baudrate
    # timeout for the serial connection
    self.timeout = timeout
    # desired value to measure
    self.mode = mode

  def open(self):
    self.ser = serial.Serial(port=self.device, baudrate=self.baudrate,
        timeout=self.timeout)
    self.ser.write(b"*RST;*CLS;*OPC?\n")
    self.ser.write(b"SENS:FUNC \"" + self.mode + b"\";  \n")
    self.ser.write(b"SENS:" + self.mode + b":NPLC 2  \n")
    # ser.readline()
    self.ser.write(b"SYST:REM\n")
    self.get_data()

  def get_data(self):
    """
    Read the signal, return False if error and print 'bad serial'.
    """
    self.ser.write(b"READ?  \n")
    t = time()
    try:
      return [t,float(self.ser.readline())]
    except (serial.SerialException,ValueError):
      self.ser.flush()
      return [t,0]

  def close(self):
    """
    Close the serial port.
    """
    self.ser.close()
