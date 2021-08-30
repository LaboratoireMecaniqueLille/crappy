# coding: utf-8

from struct import pack, unpack
import time
from .actuator import Actuator
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


def convert_to_byte(number, length):
  """This functions converts decimal into bytes.

  Mandatory in order to send or read anything into/from MAC Motors registers.
  """

  # get hex byte sequence in required '\xXX\xXX', big endian format.
  encoded = pack('%s' % length, number)
  # b = bytearray(encoded, 'hex')
  c = b''
  for i in range(0, len(encoded)):
    # x = encoded[0] ^ 0xff  # get the complement to 255
    # x = pack('B', x)  # byte formalism
    # concatenate byte and complement and add it to the sequence
    c += bytes([encoded[i], encoded[i] ^ 0xFF])
  return c


def convert_to_dec(sequence):
  """This functions converts bytes into decimals.

  Mandatory in order to send or read anything into/from MAC Motors registers.
  """

  # sequence=sequence[::2] ## cut off "complement byte"
  decim = unpack('i', sequence)  # convert to signed int value
  return decim[0]


class Biotens(Actuator):
  """Open the connection, and initialise the Biotens.

  Note:
    You should only use this class to communicate with the Biotens.
  """

  def __init__(self, port='/dev/ttyUSB0', baudrate=19200):
    """Sets the instance attributes.

    Args:
      port (:obj:`str`, optional): Path to connect to the serial port.
      baudrate (:obj:`int`, optional): Set the corresponding baud rate.
    """

    Actuator.__init__(self)
    self.port = port
    self.baudrate = baudrate

  def open(self):
    self.ser = serial.Serial(self.port, baudrate=19200, timeout=0.1)
    self.clear_errors()

  def reset_position(self):
    """Actuators goes out completely, in order to set the initial position."""

    init_position = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(38, 'B') +\
          convert_to_byte(4, 'B') +\
          convert_to_byte(0, 'i') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(38, 'B') + b'\xAA\xAA'

    init_speed = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(40, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(-50, 'h') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(40, 'B') + b'\xAA\xAA'

    init_torque = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(41, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(1023, 'i') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(41, 'B') +\
          b'\xAA\xAA'

    to_init = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(37, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(0, 'h') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(37, 'B') + b'\xAA\xAA'

    self.ser.writelines([init_position, init_speed, init_torque, to_init])
    self.ser.write(b'\x52\x52\x52\xFF\x00' +
          convert_to_byte(2, 'B') +
          convert_to_byte(2, 'B') +
          convert_to_byte(12, 'h') +
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +
          convert_to_byte(2, 'B') +
          b'\xAA\xAA')
    last_position_si = 0
    position_si = 99
    time.sleep(1)
    while position_si != last_position_si:
      last_position_si = position_si
      position_si = self.get_position()
      print("position : ", position_si)
    print("init done")
    self.stop()
    # time.sleep(1)
    # initializes the count when the motors is out.
    start_position = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(10, 'B') +\
          convert_to_byte(4, 'B') +\
          convert_to_byte(0, 'i') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(10, 'B') +\
          b'\xAA\xAA'
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
    """Stop the motor."""

    command = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(0, 'h') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(2, 'B') + b'\xAA\xAA'
    self.ser.write(command)
    # return command

  def close(self):
    self.stop()
    self.ser.close()

  def clear_errors(self):
    """Clears error in motor registers."""

    command = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(35, 'B') +\
          convert_to_byte(4, 'B') +\
          convert_to_byte(0, 'i') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(35, 'B') + b'\xAA\xAA'
    self.ser.write(command)

  def set_speed(self, speed):
    """Pilot in speed mode, requires speed in `mm/min`."""

    # converts speed in motors value
    # displacement rate in mm/min, V_SOll in 1/16 encoder counts/sample.
    # 4096 encounter counts/revolution, sampling frequency = 520.8Hz,
    # screw thread=5.
    speed_soll = int(round(16 * 4096 * speed / (520.8 * 60 * 5)))
    set_speed = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(5, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(speed_soll, 'h') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(5, 'B') + b'\xAA\xAA'

    # set torque to default value 1023
    set_torque = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(7, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(1023, 'h') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(7, 'B') +\
          b'\xAA\xAA'

    # set acceleration to 10000 mm/s²
    # (default value, arbitrarily chosen, works great so far)
    asoll = int(round(16 * 4096 * 10000 / (520.8 * 520.8 * 5)))
    set_acceleration = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(6, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(asoll, 'h') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(6, 'B') + b'\xAA\xAA+'

    command = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(1, 'h') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(2, 'B') +\
          b'\xAA\xAA'

    # write every parameters in motor's registers
    self.ser.writelines([set_speed, set_torque, set_acceleration, command])

  def set_position(self, position, speed):
    """Pilot in position mode, needs speed and final position to run
    (in `mm/min` and `mm`)."""

    # conversion of position from mm into encoder's count
    position_soll = int(round(position * 4096 / 5))
    set_position = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(3, 'B') +\
          convert_to_byte(4, 'B') +\
          convert_to_byte(position_soll, 'i') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(3, 'B') + b'\xAA\xAA+'

    # converts speed in motors value
    # displacement rate in mm/min, V_SOll in 1/16 encoder counts/sample.
    # 4096 encounter counts/revolution, sampling frequency = 520.8Hz
    # screw thread=5.
    speed_soll = int(round(16 * 4096 * speed / (520.8 * 60 * 5)))
    set_speed = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(5, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(speed_soll, 'h') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(5, 'B') +\
          b'\xAA\xAA'

    # set torque to default value 1023
    set_torque = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(7, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(1023, 'h') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(7, 'B') +\
          b'\xAA\xAA'

    # set acceleration to 10000 mm/s²
    # (default value, arbitrarily chosen, works great so far)
    asoll = int(round(16 * 4096 * 10000 / (520.8 * 520.8 * 5)))
    set_acceleration = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(6, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(asoll, 'h') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(6, 'B') +\
          b'\xAA\xAA'

    command = b'\x52\x52\x52\xFF\x00' +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(2, 'B') +\
          convert_to_byte(2, 'h') +\
          b'\xAA\xAA\x50\x50\x50\xFF\x00' +\
          convert_to_byte(2, 'B') +\
          b'\xAA\xAA'

    # write every parameters in motor's registers
    self.ser.writelines([set_position, set_speed,
                        set_torque, set_acceleration, command])

  def get_position(self):
    """Reads current position.

    Returns:
      Current position of the motor.
    """

    for i in range(20):
      r = self._get_position()
      if r is not None:
        return r
      time.sleep(.01)
    raise IOError("Could not read biotens pos!")

  def _get_position(self):
    try:
      self.ser.readlines()
    except serial.SerialException:
      # print "readlines failed"
      pass
    # print "position read"
    command = b'\x50\x50\x50\xFF\x00' + convert_to_byte(10, 'B') + b'\xAA\xAA'

    self.ser.write(command)
    # time.sleep(0.01)
    # print "reading..."
    # print self.ser.inWaiting()
    position_ = self.ser.read(19)
    if len(position_) != 19:
      return None
    # print "read"
    position = position_[9:len(position_) - 2:2]
    position = convert_to_dec(position) * 5 / 4096.
    return position
