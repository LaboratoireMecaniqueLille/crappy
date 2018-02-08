# coding: utf-8

import serial
from time import sleep
from Queue import Queue
from threading import Thread,Lock
from .actuator import Actuator

ACCEL = .1 # Acceleration and deceleration times


class Oriental(Actuator):
  """
  To drive an axis with an oriental motor through a serial link
  """

  def __init__(self, baudrate=115200, port='/dev/ttyUSB0'):
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
        #print("[DEBUG] Writing",d)
        self.ser.write(d+"\n")
        a = self.ser.readlines()
        if a:
          print("DEBUG",a)
        #self.ser.readlines()

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
    self.clear_errors()
    self.write_cmd("TA "+str(ACCEL)) # Acceleration time
    self.write_cmd("TD "+str(ACCEL)) # Deceleration time

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
    self.ser.write("SSTOP\n")

  def reset(self):
    self.clear_errors()
    self.write_cmd("RESET")
    self.write_cmd("TALK{}".format(self.num_device))
    self.clear_errors()

  def set_speed(self, speed):
    speed = int(speed*10) # Conversion en mm/min VERIFIER GAIN !!
    if speed == 0:
      self.speed = 0
      self.stop()
      return
    if speed == self.speed:
      return
    sign = speed/abs(speed)
    dirchg = self.speed*sign < 0
    if dirchg:
      #print("DEBUGORIENTAL changing dir")
      self.write_cmd("SSTOP")
      sleep(ACCEL)
    self.write_cmd("VR %d"%abs(speed))
    if sign > 0:
      #print("DEBUGORIENTAL going +")
      self.write_cmd("MCP")
    else:
      #print("DEBUGORIENTAL going -")
      self.write_cmd("MCN")
    self.speed = speed

  def set_home(self):
    self.write_cmd('preset')

  def move_home(self):
    self.write_cmd('EHOME')

  def set_position(self, position, speed):
    self.write_cmd("VR {0}".format(abs(speed)))
    self.write_cmd("MA {0}".format(position))

  def get_pos(self):
    with self.lock:
      self.ser.flushInput()
      self.ser.write('PC\n')
      self.ser.readline()
      ActuatorPos = self.ser.readline()
    ActuatorPos = str(ActuatorPos)
    try:
      ActuatorPos = float(ActuatorPos[4:-3])
    except ValueError:
      print "PositionReadingError"
      return 0
    return ActuatorPos
