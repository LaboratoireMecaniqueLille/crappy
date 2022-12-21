# coding: utf-8

from typing import Union, Optional

from .actuator import Actuator
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


class Servostar(Actuator):
  """This class can drive Kollmorgen's Servostar 300 servomotor conditioner in
  position, and set it to the analog or serial driving mode.

  It communicates with the servomotor over a serial connection.
  """

  def __init__(self,
               port: str,
               baudrate: int = 38400,
               mode: str = "serial") -> None:
    """Sets the instance attributes and initializes the parent class.

    Args:
      port (:obj:`str`): Path to connect to the serial port.
      baudrate (:obj:`int`, optional): Set the corresponding baud rate.
      mode (:obj:`str`, optional): Can be `'analog'` or `'serial'`.
    """

    super().__init__()

    self._port = port
    self._mode = mode
    self._baudrate = baudrate

    self._last_pos = None

    if mode not in ('analog', 'serial'):
      raise AttributeError(f"No such mode: {mode}")

  def open(self) -> None:
    """Initializes the serial connection and sets the desired driving mode."""

    # Opening the serial connection
    self._ser = serial.Serial(self._port, baudrate=self._baudrate, timeout=2)
    self._ser.write('ANCNFG 0\r\n')

    # Setting the desired mode
    if self._mode == "analog":
      self._set_mode_analog()
    else:
      self._set_mode_serial()

    self._ser.write('EN\r\n')
    self._ser.write('MH\r\n')

  def set_position(self,
                   pos: Union[float, bool],
                   speed: Optional[float] = None) -> None:
    """Sets the target position for the motor.

    Also allows switching to the serial or analog driving mode if the target
    position is a :obj:`bool`. The acceleration and deceleration are set to
    `200`.

    Args:
      pos: The target position to reach, as a :obj:`float`. Alternatively, a
        value of :obj:`True` sets the driving mode to serial, and :obj:`False`
        sets the driving mode to analog.
      speed: The speed at which the actuator should reach its target position.
    """

    if speed is None:
      speed = 20000

    # Nothing to do if the command is the same as the previous
    if self._last_pos == pos:
      return

    # To allow setting the mode with a boolean command, so that the actuator
    # can be driven from a single generator
    if isinstance(pos, bool):
      if pos:
        self._set_mode_serial()
      else:
        self._set_mode_analog()
      return

    # The commands can only be set from Crappy in serial mode
    elif self._mode != "serial":
      self._set_mode_serial()

    # Writing the command to the actuator
    self._ser.flushInput()
    self._ser.write(f"ORDER 0 {int(pos)} {speed} 8192 200 200 0 0 0 0\r\n")
    self._ser.write("MOVE 0\r\n")

    # Saving the last command
    self._last_pos = pos

  def get_position(self) -> Optional[float]:
    """Reads and returns the current position of the motor."""

    # Requesting a position reading
    self._ser.flushInput()
    self._ser.write("PFB\r\n")

    # Reading until getting the stop sequence, or the actuator stops responding
    r = ''
    while r != "PFB\r\n":
      # Keeping only the last 5 received characters
      if len(r) == 5:
        r = r[1:]
      # Reading the next character
      r += self._ser.read()
      # Aborting if no new character could be read
      if not r:
        print("[Servostar] Timeout error! Make sure the servostar is on !")
        return

    # The next reading should give the position
    return int(self._ser.readline())

  def stop(self) -> None:
    """Sends a command for stopping the motor."""

    self._ser.write("DIS\r\n")
    self._ser.flushInput()

  def close(self) -> None:
    """Closes the serial connection."""

    self._ser.close()

  def _set_mode_serial(self) -> None:
    """Sets the driving mode to serial."""

    self._ser.flushInput()
    self._ser.write('OPMODE 8\r\n')
    self._mode = "serial"

  def _set_mode_analog(self) -> None:
    """Sets the driving mode to analog."""

    self._last_pos = None
    self._ser.flushInput()
    self._ser.write('OPMODE 1\r\n')
    self._mode = "analog"
