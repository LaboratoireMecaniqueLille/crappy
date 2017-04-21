# coding: utf-8

import serial
from .actuator import Actuator


class Oriental(Actuator):
  """
  Open both a BiotensSensor and BiotensActuator instances.
  """

  def __init__(self, baudrate=115200, port='/dev/ttyUSB0'):
    """
    Open the connection, and initialise the Biotens.

    You should always use this Class to communicate with the Biotens.

    Argrs:
        port : str, default = '/dev/ttyUSB0'
            Path to the correct serial ser.
        size : int of float, default = 30
            Initial size of your test sample, in mm.
    """
    Actuator.__init__(self)
    self.baudrate = baudrate
    self.port = port
    self.ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=0.1)
    for i in range(1,5):
      self.ser.write("TALK{0}\n".format(i))
      ret = self.ser.readlines()
      if "{0}>".format(i) in ret:
        self.num_device = i
        motors = ['A', 'B', 'C', 'D']
        print "Motor connected to port {} is {}".format(self.port, motors[i-1])
        break

  def write_cmd(self, cmd):
    self.ser.write("{0}\n".format(cmd))
    print self.ser.readlines()

  def clear_errors(self):
    self.write_cmd("ALMCLR")

  def close(self):
    self.stop()
    self.ser.close()

  def stop(self):
    self.write_cmd("SSTOP")

  def reset(self):
    self.clear_errors()
    self.write_cmd("RESET")
    self.write_cmd("TALK{}".format(self.num_device))
    self.clear_errors()

  def set_speed(self, speed):
    if speed < 0:
      self.write_cmd("VR {0}".format(abs(speed)))
      self.write_cmd("MCP")
    if speed > 0:
      self.write_cmd("VR {0}".format(abs(speed)))
      self.write_cmd("MCN")
    if speed == 0:
      self.write_cmd("SSTOP")

  def set_home(self):
    self.write_cmd('preset')

  def move_home(self):
    self.write_cmd('EHOME')

  def set_position(self, position, speed):
    self.write_cmd("VR {0}".format(abs(speed)))
    self.write_cmd("MA {0}".format(position))

  def get_position(self):
    # self.ser.open()
    self.ser.flushInput()
    self.ser.write('PC\n')
    self.ser.readline()
    ActuatorPos = self.ser.readline()
    # self.ser.close()
    ActuatorPos = str(ActuatorPos)
    ActuatorPos = ActuatorPos[4::]
    ActuatorPos = ActuatorPos[::-1]
    ActuatorPos = ActuatorPos[3::]
    ActuatorPos = ActuatorPos[::-1]
    try:
      ActuatorPos = float(ActuatorPos)
      return ActuatorPos
    except ValueError:
      print "PositionReadingError"
