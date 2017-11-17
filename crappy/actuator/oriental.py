# coding: utf-8

import serial
from Queue import Queue
from threading import Thread,Lock
from .actuator import Actuator


class Oriental(Actuator):
  """
  To drive an axis with an oriental motor through a serial link
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
    self.speed = 0

  def reader(self):
    while True:
      d = self.q.get()
      #print("DEBUG qsize=",self.q.qsize())
      if d == None:
        break
      with self.lock:
        self.ser.write(d+"\n")
        self.ser.readlines()
        self.ser.readlines()

  def open(self):
    self.ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=0.1)
    for i in range(1,5):
      self.ser.write("TALK{0}\n".format(i))
      ret = self.ser.readlines()
      if "{0}>".format(i) in ret:
        self.num_device = i
        motors = ['A', 'B', 'C', 'D']
        print "Motor connected to port {} is {}".format(self.port, motors[i-1])
        break
    self.q = Queue()
    self.lock = Lock()
    self.reader_thread = Thread(target=self.reader)
    self.reader_thread.start()

  def write_cmd(self, cmd):
    self.q.put(cmd)

  def clear_errors(self):
    self.write_cmd("ALMCLR")

  def close(self):
    while self.q.qsize():
      self.q.get(False)
    self.q.put(None)
    self.reader_thread.join()
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
    if speed != 0 and speed == self.speed:
      return
    if speed < 0:
      if self.speed > 0:
        self.write_cmd("SSTOP")
      self.write_cmd("VR {0}".format(abs(speed)))
      self.write_cmd("MCP")
    elif speed > 0:
      if self.speed < 0:
        self.write_cmd("SSTOP")
      self.write_cmd("VR {0}".format(abs(speed)))
      self.write_cmd("MCN")
    elif speed == 0:
      self.write_cmd("SSTOP")
    self.speed = speed

  def set_home(self):
    self.write_cmd('preset')

  def move_home(self):
    self.write_cmd('EHOME')

  def set_position(self, position, speed):
    self.write_cmd("VR {0}".format(abs(speed)))
    self.write_cmd("MA {0}".format(position))

  def get_pos(self):
    # self.ser.open()
    with self.lock:
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
    except ValueError:
      print "PositionReadingError"
      return 0
    return ActuatorPos
