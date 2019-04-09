# coding: utf-8

import serial
from time import sleep
from queue import Queue
from .actuator import Actuator

ACCEL = b'.1' # Acceleration and deceleration times


class Oriental(Actuator):
  """
  To drive an axis with an oriental motor through a serial link
  """

  def __init__(self, baudrate=115200, port='/dev/ttyUSB0',gain=1/.07):
    """
    The current setup moves at .07mm/min with "VR 1"
    """
    Actuator.__init__(self)
    self.baudrate = baudrate
    self.port = port
    self.speed = 0
    self.gain = gain # unit/(mm/min)

  def open(self):
    self.ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=0.1)
    for i in range(1,5):
      self.ser.write("TALK{}\n".format(i).encode('ASCII'))
      ret = self.ser.readlines()
      if "{0}>".format(i).encode('ASCII') in ret:
        self.num_device = i
        motors = ['A', 'B', 'C', 'D']
        print("Motor connected to port {} is {}".format(self.port,motors[i-1]))
        break
    self.clear_errors()
    self.ser.write(b"TA "+ACCEL+b'\n') # Acceleration time
    self.ser.write(b"TD "+ACCEL+b'\n') # Deceleration time

  def clear_errors(self):
    self.ser.write(b"ALMCLR\n")

  def close(self):
    self.stop()
    self.ser.close()

  def stop(self):
    self.ser.write(b"SSTOP\n")
    sleep(float(ACCEL))
    #sleep(1)
    self.speed = 0

  def reset(self):
    self.clear_errors()
    self.ser.write(b"RESET\n")
    self.ser.write("TALK{}\n".format(self.num_device).encode('ASCII'))
    self.clear_errors()

  def set_speed(self, cmd):
    # speed in mm/min
    # gain can be edited by giving gain=xx to the init
    speed = min(100,int(abs(cmd*self.gain)+.5)) # Closest value < 100
    # These motors take ints only
    if speed == 0:
      self.stop()
      return
    sign = int(self.gain*cmd/abs(self.gain*cmd))
    signed_speed = sign*speed
    if signed_speed == self.speed:
      return
    dirchg = self.speed*sign < 0
    if dirchg:
      #print("DEBUGORIENTAL changing dir")
      self.stop()
    self.ser.write("VR {}\n".format(abs(speed)).encode('ASCII'))
    if sign > 0:
      #print("DEBUGORIENTAL going +")
      self.ser.write(b"MCP\n")
    else:
      #print("DEBUGORIENTAL going -")
      self.ser.write(b"MCN\n")
    self.speed = signed_speed

  def set_home(self):
    self.ser.write(b'preset\n')

  def move_home(self):
    self.ser.write(b'EHOME\n')

  def set_position(self, position, speed):
    self.ser.write("VR {0}".format(abs(speed)).encode('ASCII'))
    self.ser.write("MA {0}".format(position).encode('ASCII'))

  def get_pos(self):
    with self.lock:
      self.ser.flushInput()
      self.ser.write(b'PC\n')
      self.ser.readline()
      ActuatorPos = self.ser.readline()
    ActuatorPos = str(ActuatorPos)
    try:
      ActuatorPos = float(ActuatorPos[4:-3])
    except ValueError:
      print("PositionReadingError")
      return 0
    return ActuatorPos
