# coding: utf-8

from struct import pack, unpack
from typing import Optional
from time import sleep
import logging

from .meta_actuator import Actuator
from .._global import OptionalModule

try:
  import serial
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("pyserial")


cmd_header = b'\x52\x52\x52\xFF\x00'
cmd_tail = b'\xAA\xAA\x50\x50\x50\xFF\x00'
msg_tail_last = b'\xAA\xAA'
msg_tail_not_last = b'\xAA\xAA+'


class JVLMac140(Actuator):
  """This class allows driving JVL's MAC140 integrated servomotor in speed or
  in position.

  It interfaces with the servomotor over a serial connection.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed class from Biotens to JVLMac140
  """

  def __init__(self, port: str = '/dev/ttyUSB0') -> None:
    """Initializes the parent class.

    Args:
      port: Path to the serial port to use for communication.

    .. versionremoved:: 2.0.0 *baudrate* argument
    """

    self._ser = None

    super().__init__()

    self._port = port

  def open(self) -> None:
    """Initializes the serial connection and clears any serial error."""

    self.log(logging.INFO, f"Opening the serial port {self._port} with "
                           f"baudrate 19200")
    self._ser = serial.Serial(self._port, baudrate=19200, timeout=0.1)
    # Clearing any error in the motor registers
    cmd = self._make_cmd((35, 4, 0, 35), ('B', 'B', 'i', 'B'), True)
    self.log(logging.DEBUG, f"Writing {cmd} to port {self._port}")
    self._ser.write(cmd)

  def set_speed(self, speed: float) -> None:
    """Sets the desired speed on the actuator.

    Args:
      speed: The target speed, in `mm/min`.
    """

    # For the conversions, there are 4096 counts/motor revolution, 1/16 encoder
    # counts/sample, and the screw thread is 5
    speed = int(round(16 * 4096 * speed / (520.8 * 60 * 5)))
    acc = int(round(16 * 4096 * 10000 / (520.8 * 520.8 * 5)))

    # Generating the commands to send
    # The torque is set to 1023, the acceleration to 10000mm/s²
    set_speed = self._make_cmd((5, 2, speed, 5), ('B', 'B', 'h', 'B'), True)
    set_torque = self._make_cmd((7, 7, 1023, 7), ('B', 'B', 'h', 'B'), True)
    set_acc = self._make_cmd((6, 2, acc, 6), ('B', 'B', 'h', 'B'), False)
    command = self._make_cmd((2, 2, 1, 2), ('B', 'B', 'h', 'B'), True)

    # Writing the command values to the motor registers
    cmd = [set_speed, set_torque, set_acc, command]
    self.log(logging.DEBUG, f"Writing {cmd} to port {self._port}")
    self._ser.writelines(cmd)

  def set_position(self, position: float, speed: Optional[float]) -> None:
    """Sets the desired target position on the servomotor.

    Args:
      position: The target position, in `mm`.
      speed: The target speed for reaching the desired position, in `mm/min`.
        The speed must be given, otherwise an exception is raised.

        .. versionchanged:: 2.0.0 *speed* is now a mandatory argument
    """

    if speed is None:
      raise ValueError("The JVLMac140 actuator needs both a position and a "
                       "speed command when driven in position mode !")

    # For the conversions, there are 4096 counts/motor revolution, 1/16 encoder
    # counts/sample, and the screw thread is 5
    pos = int(round(position * 4096 / 5))
    speed = int(round(16 * 4096 * speed / (520.8 * 60 * 5)))
    acc = int(round(16 * 4096 * 10000 / (520.8 * 520.8 * 5)))

    # Generating the commands to send
    # The torque is set to 1023, the acceleration to 10000mm/s²
    set_position = self._make_cmd((3, 4, pos, 3), ('B', 'B', 'i', 'B'), False)
    set_speed = self._make_cmd((5, 2, speed, 5), ('B', 'B', 'h', 'B'), True)
    set_torque = self._make_cmd((7, 7, 1023, 7), ('B', 'B', 'h', 'B'), True)
    set_acc = self._make_cmd((6, 2, acc, 6), ('B', 'B', 'h', 'B'), True)
    command = self._make_cmd((2, 2, 2, 2), ('B', 'B', 'h', 'B'), True)

    # Writing the command values to the motor registers
    cmd = [set_position, set_speed, set_torque, set_acc, command]
    self.log(logging.DEBUG, f"Writing {cmd} to port {self._port}")
    self._ser.writelines(cmd)

  def get_position(self) -> float:
    """Reads and returns the current position of the servomotor, in `mm`.

    .. versionchanged:: 1.5.2 renamed from *get_pos* to *get_position*
    """

    # We have 20 attempts for reading the position
    for _ in range(20):
      try:
        # Emptying the read buffer
        self._ser.readlines()
        # Sending command to return position
        cmd = b''.join((b'\x50\x50\x50\xFF\x00', self._to_bytes(10, 'B'),
                        msg_tail_last))
        self.log(logging.DEBUG, f"Writing {cmd} to port {self._port}")
        self._ser.write(cmd)
        # Reading the position
        position = self._ser.read(19)
        self.log(logging.DEBUG, f"Read {position} from port {self._port}")
        # Might return fewer characters than expected due to the timeout
        if len(position) != 19:
          continue
        # Parsing the position value and returning it
        return unpack('i', position[9:17:2])[0] * 5 / 4096.
      # Catching serial errors
      except serial.SerialException:
        pass
      sleep(0.1)

    # In case no value was received after 20 attempts
    raise IOError("Could not read the position for the JVLMac140 actuator!")

  def stop(self) -> None:
    """Sends a command for stopping the servomotor."""

    if self._ser is not None:
      cmd = self._make_cmd((2, 2, 0, 2), ('B', 'B', 'h', 'B'), True)
      self.log(logging.DEBUG, f"Writing {cmd} to port {self._port}")
      self._ser.write(cmd)

  def close(self) -> None:
    """Closes the serial connection to the servomotor."""

    if self._ser is not None:
      self.log(logging.INFO, f"Closing the serial port {self._port}")
      self._ser.close()

  def reset_position(self) -> None:
    """Makes the servomotor reach its limit position, in order to re-calibrate
    the position readout."""

    init_pos = self._make_cmd((38, 4, 0, 38), ('B', 'B', 'i', 'B'), True)
    init_speed = self._make_cmd((40, 2, -50, 40), ('B', 'B', 'h', 'B'), True)
    init_torque = self._make_cmd((41, 2, 1023, 41), ('B', 'B', 'i', 'B'), True)
    to_init = self._make_cmd((37, 2, 0, 37), ('B', 'B', 'h', 'B'), True)

    cmd = [init_pos, init_speed, init_torque, to_init]
    self.log(logging.DEBUG, f"Writing {cmd} to port {self._port}")
    self._ser.writelines(cmd)
    cmd = self._make_cmd((2, 2, 12, 2), ('B', 'B', 'h', 'B'), True)
    self.log(logging.DEBUG, f"Writing {cmd} to port {self._port}")
    self._ser.write(cmd)

    sleep(1)

    last_pos = 0
    pos = 99

    # Loop while the actuator is still moving
    while pos != last_pos:
      last_pos = pos
      pos = self.get_position()
      self.log(logging.INFO, f"Current position: {pos}")
    self.log(logging.INFO, "Initialization done")
    self.stop()

    cmd = self._make_cmd((10, 4, 0, 10), ('B', 'B', 'i', 'B'), True)
    self.log(logging.DEBUG, f"Writing {cmd} to port {self._port}")
    self._ser.write(cmd)

    # Emptying the serial read buffer
    try:
      self._ser.readlines()
    except serial.SerialException:
      pass

  def _make_cmd(self,
                values: tuple[int, int, int, int],
                encodings: tuple[str, str, str, str],
                last_cmd: bool) -> bytes:
    """Builds a command to send to the servomotor, from the given arguments.

    This method is meant to simplify the code in the main methods of the class.
    """

    return b''.join((cmd_header,
                     self._to_bytes(values[0], encodings[0]),
                     self._to_bytes(values[1], encodings[1]),
                     self._to_bytes(values[2], encodings[2]),
                     cmd_tail,
                     self._to_bytes(values[3], encodings[3]),
                     msg_tail_last if last_cmd else msg_tail_not_last))

  @staticmethod
  def _to_bytes(value: float, encoding: str) -> bytes:
    """Generates bytes carrying a given value with the given encoding, and
    following the correct syntax for communicating with the servomotor."""

    encoded = pack(encoding, value)
    return b''.join((bytes((enc, enc ^ 0xFF)) for enc in encoded))
