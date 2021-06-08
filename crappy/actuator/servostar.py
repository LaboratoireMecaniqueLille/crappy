# coding: utf-8

from multiprocessing import Lock

from .actuator import Actuator
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


class Servostar(Actuator):
  """To drive and configure a servostar variator through a serial
  connection."""

  def __init__(self, device, baudrate=38400, mode="serial"):
    """Sets the instance attributes.

    Args:
      device (:obj:`str`, optional): Path to connect to the serial port.
      baudrate (:obj:`int`, optional): Set the corresponding baud rate.
      mode (:obj:`str`, optional): Can be `'analog'` or `'serial'`.
    """

    Actuator.__init__(self)
    self.devname = device
    self.mode = mode
    self.baud = baudrate
    self.lock = Lock()
    self.last = None

  def open(self):
    self.lock.acquire()
    self.ser = serial.Serial(self.devname, baudrate=self.baud, timeout=2)
    self.ser.flushInput()
    self.ser.write('ANCNFG 0\r\n')
    self.lock.release()
    if self.mode == "analog":
      self.set_mode_analog()
    elif self.mode == "serial":
      self.set_mode_serial()
    else:
      raise AttributeError("No such mode: " + str(self.mode))
    self.lock.acquire()
    self.ser.write('EN\r\n')
    self.ser.write('MH\r\n')
    self.lock.release()

  def set_position(self, pos, speed=20000, acc=200, dec=200):
    """Go to the position specified at the given speed and acceleration."""

    if self.last is pos:
      return
    if isinstance(pos, bool):
      # To use set_position(True) as set_mode_serial()
      # and set_position(False) as set_mode_analog()
      # (to command all of this from as single generator)
      if pos:
        self.set_mode_serial()
      else:
        self.set_mode_analog()
    elif self.mode != "serial":
      self.set_mode_serial()
    self.lock.acquire()
    self.ser.flushInput()
    self.ser.write(" ".join(["ORDER 0", str(int(pos)), str(speed),
                      "8192", str(acc), str(dec), "0 0 0 0\r\n"]))
    self.ser.write("MOVE 0\r\n")  # activates the order
    self.lock.release()
    self.last = pos

  def get_position(self):
    """Reads current position.

    Returns:
      Current position of the motor.
    """

    self.lock.acquire()
    self.ser.flushInput()
    self.ser.write("PFB\r\n")
    r = ''
    while r != "PFB\r\n":
      if len(r) == 5:
        r = r[1:]
      r += self.ser.read()
      if not r:
        print("Servostar timeout error! make sure the servostar is on!")
        self.lock.release()
        return
    r = ''
    while "\n" not in r:
      r += self.ser.read()
    self.lock.release()
    return int(r)

  def set_mode_serial(self):
    """Sets the serial input as setpoint."""

    self.lock.acquire()
    self.ser.flushInput()
    self.ser.write('OPMODE 8\r\n')
    self.lock.release()
    self.mode = "serial"

  def set_mode_analog(self):
    """Sets the analog input as setpoint."""

    self.last = None
    self.lock.acquire()
    self.ser.flushInput()
    self.ser.write('OPMODE 1\r\n')
    self.lock.release()
    self.mode = "analog"

  def clear_errors(self):
    """Clears error in motor registers."""

    self.ser.flushInput()
    self.ser.write("CLRFAULT\r\n")

  def stop(self):
    """Stops the motor."""

    self.ser.write("DIS\r\n")
    self.ser.flushInput()

  def close(self):
    self.ser.close()
