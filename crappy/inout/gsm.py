# coding: utf-8

import time

from .inout import InOut
from .._global import OptionalModule

try:
  import serial
  from serial.serialutil import SerialException
except (ModuleNotFoundError, ImportError):
  serial = OptionalModule("serial", "Please install the module serial to use "
      "the Gsm InOut: pip install pyserial")


class Gsm(InOut):
  """Block for sending any messages by SMS to phone numbers.

  Note:
    This block have to be associated with a modifier to manage
    which message should be sent.
  """

  def __init__(self, numbers: list = None, port: str = "/dev/ttyUSB0",
               baudrate: int = 115200):
    """Checks arguments validity.

    Args:
      numbers(:obj:`list`): A list of numbers to contact.
        ::
          syntax : "0611223344"
      port (:obj:`str`,optional): Serial port of the GSM.
      baudrate(:obj:`int`, optional): Baudrate between 1200 and 115200.
    """

    super().__init__()
    try:
      self.ser = serial.Serial(port, baudrate)
    except SerialException:
      raise SerialException("GSM not connected or wrong port")

    if numbers is None:
      raise ValueError("numbers should not be None")
    else:
      self.numbers = numbers

    # Change the type of numbers to bytes rather than string
    self.numbers = [number.encode('utf-8') for number in self.numbers]

  def open(self) -> None:
    """Call the ``_is_connected()`` method."""

    self._is_connected()

  def set_cmd(self, *cmd) -> None:
    """Send SMS to all phone numbers if not equal to "". """

    if not isinstance(cmd[0], str):
      raise TypeError("Message should be a string")
    if cmd[0] != "":
      self._send_mess(cmd[0])

  def _send_mess(self, message: str) -> None:
    print(self.numbers)
    for number in self.numbers:
      data = ""
      num = 0
      self.ser.write(b'AT' + b'\r\n')
      w_buff = [b"AT+CMGF=1\r\n",
            b"AT+CMGS=\"" + number + b"\"\r\n", message.encode()]
      while num <= 2:
        while self.ser.inWaiting() > 0:
          data += self.ser.read(self.ser.inWaiting()).decode()
          # Get all the answers in Waiting
        if data != "":
          if num < 2:
            time.sleep(1)
            self.ser.write(w_buff[num])
            # Put the message in text mode then enter the
            # number to contact
          if num == 2:
            time.sleep(0.5)
            self.ser.write(w_buff[2])  # Write the message
            self.ser.write(b"\x1a\r\n")
            # 0x1a : send   0x1b : Cancel send
          num += 1
          data = ""

  def _is_connected(self) -> None:
    """Send "AT" to the GSM and wait for an response : "OK". """

    self.ser.write(b'AT' + b'\r\n')
    data = ""
    num = 0
    while num < 2:
      while self.ser.inWaiting() > 0:
        data += self.ser.read(self.ser.inWaiting()).decode()
      if data != "":
        print(data)
        num = num + 1
        data = ""

  def close(self) -> None:
    """Close the serial port"""

    self.ser.close()
