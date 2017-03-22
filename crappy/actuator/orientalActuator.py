#!/usr/bin/python
# -*- coding: utf-8 -*-
import serial

from crappy.actuator import motion


class SerialPortActuator:
  def __init__(self, port, num_device, ConversionFactor):
    self.num_device = num_device
    print "commandcegitab !"
    # Actuator _ Declaration
    try:
      self.port = serial.Serial(port)
      self.port.timeout = 0.01
      self.port.baudrate = 115200
      self.port.bytesize = 8
      self.port.stopbits = 1
      self.port.parity = 'N'
      self.port.xonxoff = False
      self.port.rtscts = False
      self.port.dsrdtr = False
      self.port.close()
      self.port.open()
      print(self.port)
      # self.ser.close()
    except Exception as e:
      print e
    self.ConversionFactor = ConversionFactor

  def write_cmd(self, cmd):
    self.port.write("{0}\n".format(cmd))
    ret = self.port.readline()
    while ret != '{0}>'.format(self.num_device):
      print ret
      ret = self.port.readline()

  # Declaration
  def OM_Command(self, Commande):
    Commande += chr(10)
    return Commande

  # ClearAlarm
  def ClearAlarm(self):
    # self.ser.open()
    print('ALMCLR')
    self.write_cmd('ALMCLR')
    # self.ser.close()

  # MoveUp
  def MoveUp(self):
    # self.ser.open()
    print('UP')
    self.write_cmd('MCN')
    # self.ser.close()

  # MoveDown
  def MoveDown(self):
    # self.ser.open()
    print('DOWN')
    self.write_cmd('MCP')
    # self.ser.close()

  # MoveStop
  def MoveStop(self):
    # self.ser.open()
    print('STOP')
    self.write_cmd('SSTOP')
    # self.ser.close()

  # Speed (conversion from step/s to mm/s)
  def Speed(self, Speed):
    SpeedInc = float(Speed) / self.ConversionFactor
    Commande = 'VR ' + str(SpeedInc)
    # self.ser.open()
    print('ActuatorRealSpeed', Speed)
    self.write_cmd(Commande)
    # self.ser.close()

  # Position (conversion from step/s to mm/s)
  def Position(self):
    # self.ser.open()
    self.port.flushInput()
    self.port.write('PC\n')
    a_jeter = self.port.readline()
    ActuatorPos = self.port.readline()
    # self.ser.close()
    ActuatorPos = str(ActuatorPos)
    ActuatorPos = ActuatorPos[4::]
    ActuatorPos = ActuatorPos[::-1]
    ActuatorPos = ActuatorPos[3::]
    ActuatorPos = ActuatorPos[::-1]
    try:
      ActuatorPos = float(ActuatorPos) * self.ConversionFactor
      return ActuatorPos
    except ValueError:
      print "PositionReadingError"  # Se prévenir des erreur de lecture quand la readline fait de la merde

  def TarePosition(self):
    # self.ser.open()
    print('TarePosition')
    self.write_cmd('preset')
    # self.ser.close()


class OrientalActuator(motion.MotionActuator):
  def __init__(self, port='/dev/ttyUSB0', baudrate=115200, num_device=1, conversion_factor=1, ser=None):
    super(OrientalActuator, self).__init__()
    self.baudrate = baudrate
    self.num_device = num_device
    self.port = port
    # Actuator _ Declaration
    try:
      if ser is not None:
        self.ser = ser
      else:
        self.ser = serial.Serial(self.port)
        self.ser.timeout = 0.01
        self.ser.baudrate = self.baudrate
        self.ser.bytesize = 8
        self.ser.stopbits = 1
        self.ser.parity = 'N'
        self.ser.xonxoff = False
        self.ser.rtscts = False
        self.ser.dsrdtr = False
        self.ser.close()
        self.ser.open()
        for i in range(4):
          self.ser.write("TALK{0}\n".format(i + 1))
          ret = self.ser.readlines()
          if "{0}>".format(i + 1) in ret:
            self.num_device = i + 1
            motors = ['A', 'B', 'C', 'D']
            print "Motor connected to port {0} is {1}".format(self.port, motors[i])
            break
    except Exception as e:
      print e
    self.conversion_factor = conversion_factor

  def new(self):
    pass

  def write_cmd(self, cmd):
    self.ser.write("{0}\n".format(cmd))
    ret = self.ser.readline()
    # while ret != '{0}>'.format(self.num_device):
    while ret != '' and ret != '{0}>'.format(self.num_device):
      print ret
      ret = self.ser.readline()

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
    self.write_cmd("MA {0}".format(position / self.conversion_factor))

  def get_position(self):
    # self.ser.open()
    self.ser.flushInput()
    self.ser.write('PC\n')
    a_jeter = self.ser.readline()
    ActuatorPos = self.ser.readline()
    # self.ser.close()
    ActuatorPos = str(ActuatorPos)
    ActuatorPos = ActuatorPos[4::]
    ActuatorPos = ActuatorPos[::-1]
    ActuatorPos = ActuatorPos[3::]
    ActuatorPos = ActuatorPos[::-1]
    try:
      ActuatorPos = float(ActuatorPos) * self.conversion_factor
      return ActuatorPos
    except ValueError:
      print "PositionReadingError"
