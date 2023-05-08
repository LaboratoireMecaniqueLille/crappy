# coding: utf-8

from time import sleep
from typing import List
import logging

from .meta_inout import InOut
from .._global import OptionalModule

try:
  from serial import Serial
  from serial.serialutil import SerialException
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("serial", "Please install the module serial to use "
                          "the Sim868 InOut: pip install pyserial")


class Sim868(InOut):
  """This class can drive a SIM868 cellular module so that it sends SMS to
  given phone numbers.

  Important:
    This InOut should be associated with a :class:`~crappy.modifier.Modifier`
    to manage the messages to send.
  """

  def __init__(self,
               numbers: List[str],
               port: str = "/dev/ttyUSB0",
               baudrate: int = 115200) -> None:
    """Checks the validity of the arguments.

    Args:
      numbers: The list of numbers the messages will be sent to. The syntax is
        the following :
        ::

          ["0611223344"]

      port: Serial port the Sim868 is connected to.
      baudrate: Serial baudrate, between `1200` and `115200`.
    """

    self._ser = None

    super().__init__()

    self._port = port
    self._baudrate = baudrate

    # Change the type of numbers to bytes rather than string
    self._numbers = [number.encode('utf-8') for number in numbers]

  def open(self) -> None:
    """Sends ``"AT"`` to the Sim868 and waits for the response : ``"OK"``. """

    try:
      self.log(logging.INFO, f"Opening the serial port {self._port} with "
                             f"baudrate {self._baudrate}")
      self._ser = Serial(self._port, self._baudrate)
    except SerialException:
      raise SerialException("Sim868 not connected or wrong port")

    self.log(logging.DEBUG, f"Writing b'AT\\r\\n' to port {self._port}")
    self._ser.write(b'AT' + b'\r\n')
    count = 0
    while count <= 2:
      sleep(0.1)
      data = ""
      while self._ser.inWaiting() > 0:
        data += self._ser.read(self._ser.inWaiting()).decode()
      self.log(logging.DEBUG, f"Read {data} from port {self._port}")
      if "OK" in data:
        return
      count += 1
    raise TimeoutError("Sim868 is not responding")

  def set_cmd(self, *cmd: str) -> None:
    """Sends an SMS whose text is the :obj:`str` received as command to all the
    phone numbers.

    Doesn't send anything if the string is empty, and raises a :exc:`TypeError`
    if the command is not a :obj:`str`.
    """

    if not isinstance(cmd[0], str):
      raise TypeError("Message should be a string")
    if cmd[0] != "":
      self._send_mess(cmd[0])

  def _send_mess(self, message: str) -> None:
    """Commands the Sim868 to send a message to all the phone numbers.

    Args:
      message: The text message to send, as a :obj:`str`.
    """

    for number in self._numbers:
      count = 0
      self.log(logging.DEBUG, f"Writing b'AT\\r\\n' to port {self._port}")
      self._ser.write(b'AT' + b'\r\n')
      w_buff = [b"AT+CMGF=1\r\n",
                b"AT+CMGS=\"" + number + b"\"\r\n", message.encode()]
      while count <= 2:
        data = ""
        while self._ser.inWaiting() > 0:
          data += self._ser.read(self._ser.inWaiting()).decode()
          # Get all the answers in Waiting
        if data:
          self.log(logging.DEBUG, f"Read {data} from port {self._port}")
          if count < 2:
            sleep(1)
            self.log(logging.DEBUG, f"Writing {w_buff[count]} to port "
                                    f"{self._port}")
            self._ser.write(w_buff[count])
            # Put the message in text mode then enter the
            # number to contact
          if count == 2:
            sleep(0.5)
            self.log(logging.DEBUG, f"Writing {w_buff[2]} to port "
                                    f"{self._port}")
            self._ser.write(w_buff[2])  # Write the message
            self.log(logging.DEBUG, f"Writing b'\x1a\\r\\n' to port "
                                    f"{self._port}")
            self._ser.write(b"\x1a\r\n")
            # 0x1a : send   0x1b : Cancel send
          count += 1

  def close(self) -> None:
    """Closes the serial port."""

    if self._ser is not None:
      self.log(logging.INFO, f"Closing the serial port {self._port}")
      self._ser.close()
