# coding: utf-8
##  @addtogroup technical
# @{

##  @defgroup Biaxe Biaxe
# @{

## @file _biaxeTechnical.py
# @brief  Declare a new axis for the Biaxe
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 29/06/2016

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

    self.ser = serial.Serial(self.port, self.baudrate,
                             serial.EIGHTBITS, serial.PARITY_EVEN
                             , serial.STOPBITS_ONE, self.timeout)
    self.ser.write("OPMODE 0\r\n EN\r\n")

  def stop(self):
    self.ser.write("J 0\r\n")

  def reset(self):
    # TODO
    pass

  def close(self):
    """Close the designated port"""
    self.set_speed(0)
    self.stop()
    self.ser.close()

  def clear_errors(self):
    """Reset errors"""
    self.ser.write("CLRFAULT\r\n")
    self.ser.write("OPMODE 0\r\n EN\r\n")

  def set_speed(self, speed):
    """Re-define the speed of the motor. 1 = 0.002 mm/s"""
    # here we should add the physical conversion for the speed
    self.ser.write("J " + str(speed) + "\r\n")

  def set_position(self, position, speed, motion_type='relative'):
    """
    Go to a defined position with a defined speed.

    \todo
        - implement set_position, with eventually a motion_type mode
          which can be 'relative' or 'absolute'. (from actual position or from zero).
    """
    pass

  def move_home(self):
    """
    Go to position zero.

    \todo
        - implement move_home method: Go to the position zero.
    """
    pass

  def get_position(self):
    """
    return the position of the motor.

    \todo
        - implement get_position: search for the physical position of the motor.
    """
    pass
