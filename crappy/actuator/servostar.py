# coding: utf-8

from multiprocessing import Lock
from typing import Union
from warnings import warn

from .actuator import Actuator
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


class Servostar(Actuator):
  """To drive and configure a servostar variator through a serial
  connection."""

  def __init__(self,
               device: str,
               baudrate: int = 38400,
               mode: str = "serial") -> None:
    """Sets the instance attributes.

    Args:
      device (:obj:`str`): Path to connect to the serial port.
      baudrate (:obj:`int`, optional): Set the corresponding baud rate.
      mode (:obj:`str`, optional): Can be `'analog'` or `'serial'`.
    """

    warn("The Servostar Actuator will be renamed to ServoStar300 in version "
         "2.0.0", FutureWarning)

    Actuator.__init__(self)
    self.devname = device
    self.mode = mode
    self.baud = baudrate
    self.lock = Lock()
    self.last = None

  def open(self) -> None:
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

  def set_position(self,
                   pos: float,
                   speed: float = 20000,
                   acc: float = 200,
                   dec: float = 200) -> None:
    """Go to the position specified at the given speed and acceleration."""

    warn("The speed argument of set_position will not be optional anymore in "
         "version 2.0.0, and will be None if no speed is set", FutureWarning)
    warn("The acc and dec arguments of set_position will be removed in "
         "version 2.0.0", FutureWarning)

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

  def get_position(self) -> Union[float, None]:
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

  def set_mode_serial(self) -> None:
    """Sets the serial input as setpoint."""

    warn("The set_mode_serial method will be renamed to _set_mode_serial in "
         "version 2.0.0", FutureWarning)

    self.lock.acquire()
    self.ser.flushInput()
    self.ser.write('OPMODE 8\r\n')
    self.lock.release()
    self.mode = "serial"

  def set_mode_analog(self) -> None:
    """Sets the analog input as setpoint."""

    warn("The set_mode_analog method will be renamed to _set_mode_analog in "
         "version 2.0.0", FutureWarning)

    self.last = None
    self.lock.acquire()
    self.ser.flushInput()
    self.ser.write('OPMODE 1\r\n')
    self.lock.release()
    self.mode = "analog"

  def clear_errors(self) -> None:
    """Clears error in motor registers."""

    warn("The clear_errors method will be removed in version 2.0.0",
         FutureWarning)

    self.ser.flushInput()
    self.ser.write("CLRFAULT\r\n")

  def stop(self) -> None:
    """Stops the motor."""

    self.ser.write("DIS\r\n")
    self.ser.flushInput()

  def close(self) -> None:
    self.ser.close()
