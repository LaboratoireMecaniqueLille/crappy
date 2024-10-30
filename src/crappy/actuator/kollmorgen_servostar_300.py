# coding: utf-8

from typing import Union, Optional, Literal
import logging

from .meta_actuator import Actuator
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


class ServoStar300(Actuator):
  """This class can drive a Kollmorgen ServoStar 300 servomotor conditioner in
  position, and set it to the analog or serial driving mode.

  It communicates with the servomotor over a serial connection. The
  :class:`~crappy.lamcube.Biaxe` Actuator can drive the same hardware, but only
  in speed.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Servostar* to *ServoStar300*
  """

  def __init__(self,
               port: str,
               baudrate: int = 38400,
               mode: Literal['serial', 'analog'] = "serial") -> None:
    """Sets the instance attributes and initializes the parent class.

    Args:
      port: Path to the serial port used for communication.

        .. versionchanged:: renamed from *device* to *port*
      baudrate: The serial baud rate to use, as an :obj:`int`.
      mode: The driving mode to use when starting the test. Can be `'analog'`
        or `'serial'`. It can be changed afterward while the test is running,
        by sending the right command.
    """

    self._ser = None

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
    self.log(logging.INFO, f"Opening the serial port {self._port} with "
                           f"baudrate {self._baudrate}")
    self._ser = serial.Serial(self._port, baudrate=self._baudrate, timeout=2)
    self.log(logging.DEBUG, f"Writing b'ANCNFG 0\\r\\n' to port {self._port}")
    self._ser.write(b'ANCNFG 0\r\n')

    # Setting the desired mode
    if self._mode == "analog":
      self._set_mode_analog()
    else:
      self._set_mode_serial()

    self.log(logging.DEBUG, f"Writing b'EN\\r\\n' to port {self._port}")
    self._ser.write(b'EN\r\n')
    self.log(logging.DEBUG, f"Writing b'MH\\r\\n' to port {self._port}")
    self._ser.write(b'MH\r\n')

  def set_position(self,
                   pos: Union[float, bool],
                   speed: Optional[float]) -> None:
    """Sets the target position for the motor.

    Also allows switching to the serial or analog driving mode if the target
    position is a :obj:`bool`. The acceleration and deceleration are set to
    `200`.

    Args:
      pos: The target position to reach, as a :obj:`float`. Alternatively, a
        value of :obj:`True` sets the driving mode to serial, and :obj:`False`
        sets the driving mode to analog.
      speed: The speed at which the actuator should reach its target position.
        If no speed is specified, the default is `20000`.

        .. versionchanged:: 2.0.0 *speed* is now a mandatory argument
    
    .. versionremoved:: 2.0.0 *acc* and *dec* arguments
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
    self.log(logging.DEBUG, f"Writing b'ORDER 0 {int(pos)} {speed} 8192 200 "
                            f"200 0 0 0 0\\r\\n' to port {self._port}")
    self._ser.write(f"ORDER 0 {int(pos)} {speed} 8192 200 200 0 0 0 0\r\n")
    self.log(logging.DEBUG, f"Writing b'MOVE 0\\r\\n' to port {self._port}")
    self._ser.write("MOVE 0\r\n")

    # Saving the last command
    self._last_pos = pos

  def get_position(self) -> Optional[float]:
    """Reads and returns the current position of the motor.

    .. versionchanged:: 1.5.2 renamed from *get_pos* to *get_position*
    """

    # Requesting a position reading
    self._ser.flushInput()
    self.log(logging.DEBUG, f"Writing b'PFB\\r\\n' to port {self._port}")
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
        self.log(logging.ERROR, "Timeout error ! Make sure the servostar is "
                                "on !")
        return

    # The next reading should give the position
    ret = self._ser.readline()
    self.log(logging.DEBUG, f"Read {ret} on port {self._port}")
    return int(ret)

  def stop(self) -> None:
    """Sends a command for stopping the motor."""

    if self._ser is not None:
      self.log(logging.DEBUG, f"Writing b'DIS\\r\\n' to port {self._port}")
      self._ser.write("DIS\r\n")
      self._ser.flushInput()

  def close(self) -> None:
    """Closes the serial connection."""

    if self._ser is not None:
      self.log(logging.INFO, f"Closing the serial port {self._port}")
      self._ser.close()

  def _set_mode_serial(self) -> None:
    """Sets the driving mode to serial."""

    self._ser.flushInput()
    self.log(logging.DEBUG, f"Writing b'OPMODE 8\\r\\n' to port {self._port}")
    self._ser.write('OPMODE 8\r\n')
    self._mode = "serial"

  def _set_mode_analog(self) -> None:
    """Sets the driving mode to analog."""

    self._last_pos = None
    self._ser.flushInput()
    self.log(logging.DEBUG, f"Writing b'OPMODE A\\r\\n' to port {self._port}")
    self._ser.write('OPMODE 1\r\n')
    self._mode = "analog"
