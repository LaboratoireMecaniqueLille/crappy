# coding: utf-8
##  @addtogroup actuator
# @{

##  @defgroup CmDriveActuator CmDriveActuator
# @{

## @file _CmdriveActuator.py
# @brief  Open a new default serial port for communication with Servostar
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 04/07/2016

import serial
import time
from ._meta import motion
from .._warnings import deprecated as deprecated


class CmDriveActuator(motion.MotionActuator):
  """ Open a new default serial port for communication with Servostar"""

  def __init__(self, ser=None, port='/dev/serial/by-path/pci-0000:00:14.0-usb-0:6:1.0-port0', baudrate=9600):

    super(CmDriveActuator, self).__init__()
    self.port = port
    self.baudrate = baudrate
    if ser is not None:
      self.ser = ser
    else:
      self.ser = serial.Serial(self.port, self.baudrate)

  def set_speed(self, speed):
    self.ser.close()  # close serial connection before to avoid errors
    self.ser.open()  # open serial port
    # velocity = input ('Velocity: \n')#request to the user about velocity
    if abs(speed) < 1000000:
      self.ser.write(
        'SL ' + str(int(speed)) + '\r')  # send ASCII characters to the servostar to apply velocity task
      self.ser.read(self.ser.inWaiting())
    else:
      print 'Maximum speed exeeded'
    self.ser.close()  # close serial connection

  def set_position(self, position, speed, motion_type='relative'):
    self.ser.close()  # close serial connection before to avoid errors
    self.ser.open()  # open serial port

    if motion_type == 'relative':
      self.ser.write('MR %i\r' % (position))  # send ASCII characters to apply the selected motion task
    if motion_type == 'absolute':
      self.ser.write('MA %i\r' % (position))  # send ASCII characters to apply the selected motion task
    self.ser.readline()
    self.ser.close()  # close serial connection

  def move_home(self):
    """Reset the position to zero"""
    self.ser.open()  # open serial port
    self.ser.write('MA 0\r')  # send 'MH' ASCII characters for requesting to the motor to return at zero position
    # self.ser.readline()
    self.ser.close()  # close serial connection

  @deprecated(None, "connection is initialized in __init__")
  def setConnection(self, port, baudrate):
    """
    DEPRECATED: connection is initialized in __init__.
    Open a new specified serial port for communication with Servostar
    """
    # self.myPort = port
    # self.baudrate = baudrate
    # self.ser = serial.Serial(self.myPort, self.baudrate)
    # self.ser.close()
    # return self.ser
    pass

  """
  Methods controlling motion
  =============================
  """

  @deprecated(None, 'stop method is defined in _cmdriveTechnical')
  def stopMotion(self):
    """
    DEPRECATED: stop method is defined in _cmdriveTechnical.
    Stop the motor motion
    """
    self.ser.close()  # close serial connection before to avoid errors
    self.ser.open()
    self.ser.write('SL 0\r')
    self.ser.readline()
    self.ser.close()

  @deprecated(set_position)
  def applyAbsoluteMotion(self, position):
    """
    DEPRECATED: use set_position(position, None, motion_type='absolute') instead.
    Absolut displacement from zero
    """
    self.ser.close()  # close serial connection before to avoid errors
    self.ser.open()  # open serial port
    self.ser.write('MA %i\r' % (position))  # send ASCII characters to apply the selected motion task
    self.ser.readline()
    self.ser.close()  # close serial connection

  @deprecated(set_position)
  def applyRelativeMotion(self, num):
    """
    DEPRECATED: use set_position(position, None, motion_type='relative') instead.
    Relative displacement from current position
    """
    self.ser.close()  # close serial connection before to avoid errors
    self.ser.open()  # open serial port
    self.ser.write('MR %i\r' % (num))  # send ASCII characters to apply the selected motion task
    self.ser.readline()
    self.ser.close()  # close serial connection

  """Methods controlling speed
  ============================"""

  @deprecated(set_speed)
  def applyPositiveSpeed(self, speed):
    """
    DEPRECATED: use set_speed(speed) instead.
    Positive displacement at a setted speed
    """
    self.ser.close()  # close serial connection before to avoid errors
    self.ser.open()  # open serial port
    # velocity = input ('Velocity: \n') #request to the user about velocity
    if speed < 1000000:
      self.ser.write('SL %i\r' % speed)  # send ASCII characters to the servostar to apply velocity task
      self.ser.readline()
    else:
      print 'Maximum speed exeeded'
    self.ser.close()  # close serial connection

  @deprecated(set_speed)
  def applyNegativeSpeed(self, speed):
    """
    DEPRECATED: use set_speed(-speed) instead.
    Negative displacement at a setted speed
    """
    self.ser.close()  # close serial connection before to avoid errors
    self.ser.open()  # open serial port
    # velocity = input ('Velocity: \n')#request to the user about velocity
    if speed < 1000000:
      self.ser.write('SL -%i\r' % speed)  # send ASCII characters to the servostar to apply velocity task
      self.ser.readline()
    else:
      print 'Maximum speed exeeded'
    self.ser.close()  # close serial connection

  @deprecated(set_speed)
  def applyAbsoluteSpeed(self, speed):
    """
    DEPRECATED: use set_speed instead.
    Positive or Negative displacement at a setted speed
    """
    self.ser.close()  # close serial connection before to avoid errors
    self.ser.open()  # open serial port
    # velocity = input ('Velocity: \n')#request to the user about velocity
    if abs(speed) < 1000000:
      self.ser.write(
        'SL ' + str(int(speed)) + '\r')  # send ASCII characters to the servostar to apply velocity task
      self.ser.read(self.ser.inWaiting())
    else:
      print 'Maximum speed exeeded'
    self.ser.close()  # close serial connection

  """
  Methods checking & controlling position
  =======================================
  """

  @deprecated(None, 'use get_position instead, defined in CmDriveSensor')
  def examineLocation(self):
    """
    DEPRECATED: use get_position instead, defined in CmDriveSensor
    Search for the physical position of the motor
    """
    self.ser.close()
    ser = self.setConnection(self.myPort, self.baudrate)  # initialise serial port
    self.ser.open()
    self.ser.write('PR P \r')  # send 'PFB' ASCII characters to request the location of the motor
    pfb = self.ser.readline()  # read serial data from the buffer
    pfb1 = self.ser.readline()  # read serial data from the buffer
    print '%s %i' % (pfb, (int(pfb1)))  # print location
    print '\n'
    self.ser.close()  # close serial connection
    return int(pfb1)

  @deprecated(None, 'Use reset instead, defined in CmDriveTechnical')
  def resetZero(self):
    """
    DEPRECATED: Use reset instead, defined in CmDriveTechnical.
    Reset the serial communication, before reopen it to set displacement to zero
    """
    self.ser.close()  # ????????????
    self.ser = self.setConnection(self.myPort, self.baudrate)  # initialise serial port
    self.ser.open()  # open serial port
    import Tkinter
    import tkMessageBox
    result = tkMessageBox.askyesno('resetZero',
                                   'Warning! The recorded trajectories will be erased, continue?')  # send request to the user if he would reset the system
    if result is True:
      self.ser.write('DIS\r')  # send 'DIS' ASCII characters to disable the motor
      self.ser.write('SAVE\r')  # send 'SAVE' ASCII characters to SAVE servostar values
      self.ser.write('COLDSTART\r')  # send 'COLDSTART' ASCII characters to reboot servostar
      k = 0
      # print different stages of booting
      while k < 24:
        print self.ser.readline()
        k += 1
      # self.ser.close() #close serial connection
      return 1
    else:
      # self.ser.close() #close serial connection
      return 0

  @deprecated(move_home)
  def moveZero(self):
    """
    DEPRECATED: use move_home instead.
    Reset the position to zero
    """
    self.ser.open()  # open serial port
    self.ser.write('MA 0\r')  # send 'MH' ASCII characters for requesting to the motor to return at zero position
    self.ser.readline()
    self.ser.close()  # close serial connection

  @deprecated(None, 'use close method instead, defined in CmDriveTechnical')
  def close_port(self):
    """
    DEPRECATED: use close method instead, defined in CmDriveTechnical.
    Close the designated port
    """
    self.ser.close()

  @deprecated(None, "use clear_errors instead, defined in CmDriveTechnical")
  def CLRFAULT(self):
    """
    DEPRECATED: use clear_errors instead, defined in CmDriveTechnical.
    Reset errors
    """
    self.ser.write("CLRFAULT\r\n")
    self.ser.write("OPMODE 0\r\n EN\r\n")
