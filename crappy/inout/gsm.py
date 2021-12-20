# coding: utf-8

from time import sleep

from .inout import InOut
from .._global import OptionalModule

try:
  from serial import Serial
  from serial.serialutil import SerialException
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("serial", "Please install the module serial to use "
                          "the Gsm InOut: pip install pyserial")


class Gsm(InOut):
  """Block for sending messages by SMS to given phone numbers.

  Important:
    This block should be associated with a modifier to manage the messages
    to send.
  """

  def __init__(self,
               numbers: list,
               port: str = "/dev/ttyUSB0",
               baudrate: int = 115200) -> None:
    """Checks arguments validity.

    Args:
      numbers(:obj:`list`): The list of numbers the messages will be sent to.
        The syntax is the following :
        ::

          ["0611223344"]

      port (:obj:`str`, optional): Serial port the GSM is connected to.
      baudrate(:obj:`int`, optional): Serial baudrate, between 1200 and 115200.
    """

    super().__init__()
    try:
      self._ser = Serial(port, baudrate)
    except SerialException:
      raise SerialException("GSM not connected or wrong port")

    # Change the type of numbers to bytes rather than string
    self._numbers = [number.encode('utf-8') for number in numbers]

  def open(self) -> None:
    """Sends ``"AT"`` to the GSM and waits for the response : ``"OK"``. """

    self._ser.write(b'AT' + b'\r\n')
    count = 0
    while count <= 2:
      sleep(0.1)
      data = ""
      while self._ser.inWaiting() > 0:
        data += self._ser.read(self._ser.inWaiting()).decode()
      if "OK" in data:
        return
      count += 1
    raise TimeoutError("GSM is not responding")

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
    """Commands the GSM to send a message to all the phone numbers.

    Args:
      message: The text message to send.
    """

    for number in self._numbers:
      count = 0
      self._ser.write(b'AT' + b'\r\n')
      w_buff = [b"AT+CMGF=1\r\n",
                b"AT+CMGS=\"" + number + b"\"\r\n", message.encode()]
      while count <= 2:
        data = ""
        while self._ser.inWaiting() > 0:
          data += self._ser.read(self._ser.inWaiting()).decode()
          # Get all the answers in Waiting
        if data:
          if count < 2:
            sleep(1)
            self._ser.write(w_buff[count])
            # Put the message in text mode then enter the
            # number to contact
          if count == 2:
            sleep(0.5)
            self._ser.write(w_buff[2])  # Write the message
            self._ser.write(b"\x1a\r\n")
            # 0x1a : send   0x1b : Cancel send
          count += 1

  def close(self) -> None:
    """Closes the serial port."""

    self._ser.close()
