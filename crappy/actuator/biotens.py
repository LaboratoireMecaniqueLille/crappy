# coding: utf-8

from struct import pack,unpack
import serial
import time
from .actuator import Actuator


def convert_to_byte(number, length):
  """This functions converts decimal into bytes or bytes into decimals.
  Mandatory in order to send or read anything into/from MAC Motors registers"""
  # get hex byte sequence in required '\xXX\xXX', big endian format.
  encoded = pack('%s' % length, number)
  b = bytearray(encoded, 'hex')
  i = 0
  c = ''
  for i in range(0, len(encoded)):
    x = int(b[i]) ^ 0xff  # get the complement to 255
    x = pack('B', x)  # byte formalism
    # concatenate byte and complement and add it to the sequece
    c += encoded[i] + '%s' % x
  return c


def convert_to_dec(sequence):
  """
  This functions converts bytes into decimals.  Mandatory in order to send
  or read anything into/from MAC Motors registers.
  """
  # sequence=sequence[::2] ## cut off "complement byte"
  decim = unpack('i', sequence)  # convert to signed int value
  return decim[0]


class Biotens(Actuator):
  def __init__(self, port='/dev/ttyUSB0', baudrate=19200):
    """
    Open the connection, and initialise the Biotens.

    You should always use this Class to communicate with the Biotens.

    Args:
        port : str, default = '/dev/ttyUSB0'
            Path to the connect serial port.
    """
    Actuator.__init__(self)
    self.port = port
    self.baudrate = baudrate

  def open(self):
    self.ser = serial.Serial(self.port, baudrate=19200, timeout=0.1)
    self.clear_errors()

  def reset_position(self):
    """Actuators goes out completely, in order to set the initial position"""
    init_position = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(38, 'B') +\
          convert_to_byte(4, 'B') +\
          convert_to_byte(0, 'i') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(38, 'B') + '\xAA\xAA'

    init_speed = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(40, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(-50, 'h') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(40, 'B') + '\xAA\xAA'

    init_torque = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(41, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(1023, 'i') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(41, 'B') +\
          '\xAA\xAA'

    to_init = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(37, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(0, 'h') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(37, 'B') + '\xAA\xAA'

    self.ser.writelines([init_position, init_speed, init_torque, to_init])
    self.ser.write('\x52\x52\x52\xFF\x00'+
          convert_to_byte(2, 'B') +
          convert_to_byte(2, 'B') +
          convert_to_byte(12, 'h') +
          '\xAA\xAA\x50\x50\x50\xFF\x00' +
          convert_to_byte(2, 'B') +
          '\xAA\xAA')
    last_position_si = 0
    position_si = 99
    time.sleep(1)
    while position_si != last_position_si:
      last_position_si = position_si
      position_si = self.get_pos()
      print("position : ", position_si)
    print("init done")
    self.stop()
    # time.sleep(1)
    # initializes the count when the motors is out.
    start_position = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(10, 'B') +\
          convert_to_byte(4, 'B') +\
          convert_to_byte(0, 'i') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(10, 'B') +\
          '\xAA\xAA'
    self.ser.write(start_position)
    # time.sleep(1)
    try:
      self.ser.readlines()
    except serial.SerialException:
      pass

  def reset(self):
    # TODO
    pass

  def stop(self):
    """Stop the motor"""
    command = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(0, 'h') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(2, 'B') + '\xAA\xAA'
    self.ser.write(command)
    # return command

  def close(self):
    self.stop()
    self.ser.close()

  def clear_errors(self):
    """
    Clears error in motor registers. obviously.
    """
    command = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(35, 'B') +\
          convert_to_byte(4, 'B') +\
          convert_to_byte(0, 'i') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(35, 'B') + '\xAA\xAA'
    self.ser.write(command)

  def set_speed(self, speed):
    """Pilot in speed mode, requires speed in mm/min"""
    # converts speed in motors value
    # displacement rate in mm/min, V_SOll in 1/16 encoder counts/sample.
    # 4096 encounder counts/revolution, sampling frequency = 520.8Hz,
    # screw thread=5.
    speed_soll = int(round(16 * 4096 * speed / (520.8 * 60 * 5)))
    set_speed = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(5, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(speed_soll, 'h') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(5, 'B') + '\xAA\xAA'

    # set torque to default value 1023
    set_torque = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(7, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(1023, 'h') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(7, 'B') +\
          '\xAA\xAA'

    # set acceleration to 10000 mm/s²
    # (default value, arbitrarily chosen, works great so far)
    asoll = int(round(16 * 4096 * 10000 / (520.8 * 520.8 * 5)))
    set_acceleration = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(6, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(asoll, 'h') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(6, 'B') + '\xAA\xAA+'

    command = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(1, 'h') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(2, 'B') +\
          '\xAA\xAA'

    # write every parameters in motor's registers
    self.ser.writelines([set_speed, set_torque, set_acceleration, command])

  def set_position(self, position, speed):
    """Pilot in position mode, needs speed and final position to run
    (in mm/min and mm)"""
    # conversion of position from mm into encoder's count
    position_soll = int(round(position * 4096 / 5))
    set_position = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(3, 'B') +\
          convert_to_byte(4, 'B') +\
          convert_to_byte(position_soll, 'i') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(3, 'B') + '\xAA\xAA+'

    # converts speed in motors value
    # displacement rate in mm/min, V_SOll in 1/16 encoder counts/sample.
    # 4096 encounder counts/revolution, sampling frequency = 520.8Hz
    # screw thread=5.
    speed_soll = int(round(16 * 4096 * speed / (520.8 * 60 * 5)))
    set_speed = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(5, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(speed_soll, 'h') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(5, 'B') +\
          '\xAA\xAA'

    # set torque to default value 1023
    set_torque = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(7, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(1023, 'h') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(7, 'B') +\
          '\xAA\xAA'

    # set acceleration to 10000 mm/s²
    # (default value, arbitrarily chosen, works great so far)
    asoll = int(round(16 * 4096 * 10000 / (520.8 * 520.8 * 5)))
    set_acceleration = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(6, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(asoll, 'h') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(6, 'B') +\
          '\xAA\xAA'

    command = '\x52\x52\x52\xFF\x00' +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(2, 'h') +\
          '\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(2, 'B') +\
          '\xAA\xAA'

    # write every parameters in motor's registers
    self.ser.writelines([set_position, set_speed,
                        set_torque, set_acceleration, command])

  def get_pos(self):
    """
    Reads current position

    Returns:
        current position of the motor.
    """
    try:
      self.ser.readlines()
    except serial.SerialException:
      # print "readlines failed"
      pass
    # print "position read"
    command = '\x50\x50\x50\xFF\x00' + convert_to_byte(10, 'B') + '\xAA\xAA'

    self.ser.write(command)
    # time.sleep(0.01)
    # print "reading..."
    # print self.ser.inWaiting()
    position_ = self.ser.read(19)
    # print "read"
    position = position_[9:len(position_) - 2:2]
    position = convert_to_dec(position) * 5 / 4096.
    return position
