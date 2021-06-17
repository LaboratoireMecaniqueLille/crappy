# coding: utf-8

from .actuator import Actuator
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


class CM_drive(Actuator):
  """Open a new default serial port for communication with a CMdrive actuator.
  """

  def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
    """Sets the instance attributes.

    Args:
      port (:obj:`str`, optional): Path to connect to the serial port.
      baudrate (:obj:`int`, optional): Set the corresponding baud rate.
    """

    Actuator.__init__(self)
    self.port = port
    self.baudrate = baudrate

  def open(self):
    self.ser = serial.Serial(self.port, self.baudrate)

  def stop(self):
    """Stop the motor motion."""

    # close serial connection before to avoid errors
    self.ser.close()
    self.ser.open()
    self.ser.write('SL 0\r')
    # self.ser.readline()
    self.ser.close()

  def reset(self):
    """Reset the serial communication, before reopening it to set displacement
    to zero."""

    self.ser.close()
    self.ser.open()  # open serial port
    result = input("Reset the system ? This will erase recorded trajectories")
    # send request to the user if he would reset the system
    if result.lower()[0] in ['y', 'o']:
      # send 'DIS' ASCII characters to disable the motor
      self.ser.write('DIS\r')
      # send 'SAVE' ASCII characters to SAVE servostar values
      self.ser.write('SAVE\r')
      # send 'COLDSTART' ASCII characters to reboot servostar
      self.ser.write('COLDSTART\r')
      k = 0
      # print different stages of booting
      while k < 24:
        print(self.ser.readline())
        k += 1
      # self.ser.close() #close serial connection
      return 1
    else:
      # self.ser.close() #close serial connection
      return 0

  def close(self):
    """Close the designated port."""

    self.stop()
    self.ser.close()

  def clear_errors(self):
    """Reset errors."""

    self.ser.write("CLRFAULT\r\n")
    self.ser.write("OPMODE 0\r\n EN\r\n")

  def set_speed(self, speed):
    """Pilot in speed mode, requires speed in `mm/min`."""

    self.ser.close()  # close serial connection before to avoid errors
    self.ser.open()  # open serial port
    # velocity = input ('Velocity: \n')#request to the user about velocity
    if abs(speed) < 1000000:
      # send ASCII characters to the servostar to apply velocity task
      self.ser.write('SL ' + str(int(speed)) + '\r')
      self.ser.read(self.ser.inWaiting())
    else:
      print('Maximum speed exceeded')
    self.ser.close()  # close serial connection

  def set_position(self, position, speed, motion_type='relative'):
    """Pilot in position mode, needs speed and final position to run
    (in `mm/min` and `mm`)."""

    self.ser.close()  # close serial connection before to avoid errors
    self.ser.open()  # open serial port

    if motion_type == 'relative':
      # send ASCII characters to apply the selected motion task
      self.ser.write('MR %i\r' % position)
    if motion_type == 'absolute':
      # send ASCII characters to apply the selected motion task
      self.ser.write('MA %i\r' % position)
    self.ser.readline()
    self.ser.close()  # close serial connection

  def move_home(self):
    """Reset the position to zero."""

    self.ser.open()  # open serial port
    # send 'MH' ASCII characters for requesting to the motor to return at pos 0
    self.ser.write('MA 0\r')
    # self.ser.readline()
    self.ser.close()  # close serial connection

  def get_position(self):
    """Search for the physical position of the motor.

    Returns:
      Physical position of the motor.
    """

    self.ser.close()
    # ser=self.setConnection(self.myPort, self.baudrate)
    # initialise serial port
    self.ser.open()
    # send 'PFB' ASCII characters to request the location of the motor
    self.ser.write('PR P \r')
    pfb = self.ser.readline()  # read serial data from the buffer
    pfb1 = self.ser.readline()  # read serial data from the buffer
    print('%s %i' % (pfb, (int(pfb1))))  # print location
    print('\n')
    self.ser.close()  # close serial connection
    return int(pfb1)
