# coding: utf-8
import serial
from .inout import InOut


class Agilent34420A(InOut):
  """Sensor class for Agilent34420A devices."""

  def __init__(self, mode="VOLT", device='/dev/ttyUSB0', baudrate=9600, timeout=10):
    """
    This class contains method to measure values of resistance or tensution on Agilent34420A devices.

    May work for other devices too, but not tested.
    If you have issues with this class returning a lot of 'bad serial', \
    make sure you have the last version of pySerial.

    Args:
        mode : {"VOLT","RES"} , default = "VOLT"
                Desired value to measure.
        device : str, default = '/dev/ttyUSB0'
                Path to the device.
        baudrate : int, default = 9600
                Desired baudrate.
        timeout : int or float, default = 10
                Timeout for the serial connection.
    """
    InOut.__init__(self)
    ## path to the device
    self.device = device
    ## desired baudrate
    self.baudrate = baudrate
    ## timeout for the serial connection
    self.timeout = timeout
    ## desired value to measure
    self.mode = mode

  def open(self):
    self.ser = serial.Serial(port=self.device, baudrate=self.baudrate,
        timeout=self.timeout)
    self.ser.write("*RST;*CLS;*OPC?\n")
    self.ser.write("SENS:FUNC \"" + self.mode + "\";  \n")
    self.ser.write("SENS:" + self.mode + ":NPLC 2  \n")
    # ser.readline()
    self.ser.write("SYST:REM\n")
    self.get_data()

  def get_data(self):
    """
    Read the signal, return False if error and print 'bad serial'.
    """
    self.ser.write("READ?  \n")
    tmp = self.ser.read(self.ser.in_waiting)
    self.ser.flush()
    return float(tmp)

  def close(self):
    """
    Close the serial port.
    """
    self.ser.close()

