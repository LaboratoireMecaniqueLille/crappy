# coding: utf-8
import serial
from .actuator import Actuator


class Biaxe(Actuator):
  """Declare a new axis for the Biaxe"""

  def __init__(self, port='/dev/ttyUSB0', baudrate=38400, timeout=1):
    """
    This class create an axis and opens the corresponding serial port.

    Args:
        port : str
                Path to the corresponding serial port, e.g '/dev/ttyS4'
        baudrate : int, default = 38400
                Set the corresponding baud rate.
        timeout : int or float, default = 1
                Serial timeout.
    """
    Actuator.__init__(self)
    self.port = port
    self.baudrate = baudrate
    self.timeout = timeout

  def open(self):
    self.ser = serial.Serial(self.port, self.baudrate,
                             serial.EIGHTBITS, serial.PARITY_EVEN
                             , serial.STOPBITS_ONE, self.timeout)
    self.clear_errors()

  def stop(self):
    self.ser.write("J 0\r\n")

  def close(self):
    """Close the designated port"""
    self.stop()
    self.ser.close()

  def clear_errors(self):
    """Reset errors"""
    self.ser.write("CLRFAULT\r\n")
    self.ser.write("OPMODE 0\r\n EN\r\n")

  def set_speed(self, speed):
    """Re-define the speed of the motor. 1 = 0.002 mm/s"""
    speed = int(speed/.002) # Convert to mm/s
    self.ser.write("J " + str(speed) + "\r\n")
