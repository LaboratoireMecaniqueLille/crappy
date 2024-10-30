# coding: utf-8

from time import sleep, time
from typing import Optional
from collections.abc import Iterable
from re import fullmatch
import logging
from  warnings import warn

from .meta_inout import InOut
from .._global import OptionalModule

try:
  from serial import Serial
  from serial.serialutil import SerialException
except (ModuleNotFoundError, ImportError):
  Serial = OptionalModule("pyserial")
  SerialException = OptionalModule("pyserial")


class Sim868(InOut):
  """This class can drive a SIM868 cellular module so that it sends SMS to
  given phone numbers.

  Important:
    This InOut should be associated with a :class:`~crappy.modifier.Modifier`
    to manage the messages to send.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Gsm* to *Sim868*
  """

  def __init__(self,
               numbers: Iterable[str],
               port: str = "/dev/ttyUSB0",
               baudrate: int = 115200,
               pin_code: Optional[str] = None,
               registration_timeout: float = 10) -> None:
    """Checks the validity of the arguments.

    Args:
      numbers: An iterable (like a :obj:`list` or a :obj:`tuple`) of numbers
        the messages will be sent to. The syntax is the following :
        ::

          ["0611223344"]

      port: Serial port the Sim868 is connected to.
      baudrate: Serial baudrate, between `1200` and `115200`.
      pin_code: Optionally, a pin code to use for activating the SIM card.
      
        .. versionadded:: 2.0.0
      registration_timeout: The maximum number of seconds to allow for the
        Sim868 to register to a network once the SIM card has the ready status.

        .. versionadded:: 2.0.0
    """

    warn(f"Starting from version 2.1.0, {type(self).__name__} will be moved "
         f"to crappy.collection. Your code that uses it will still work as "
         f"is, except you will now need to import crappy.collection at the "
         f"top of your script.", FutureWarning)

    self._ser = None

    super().__init__()

    self._port = port
    self._baudrate = baudrate
    self._pin = pin_code
    self._reg_timeout = registration_timeout

    # Change the type of numbers to bytes rather than string
    self._numbers = list(numbers)

  def open(self) -> None:
    """Initializes the Sim868 device and checks its network connection.

    First, the serial connection is checked. Then, checking if the SIM card
    requires a PIN code. If so and if one is given, sets the PIN code on the
    SIM. Then, checks that the SIM is connected to a network. Finally, sets the
    input mode to Text for the SMS.
    """

    try:
      self.log(logging.INFO, f"Opening the serial port {self._port} with "
                             f"baudrate {self._baudrate}")
      self._ser = Serial(self._port, self._baudrate, timeout=0)
    except SerialException:
      raise SerialException("Sim868 not connected or wrong port")

    # Polling the Sim868 and waiting for a response
    self.log(logging.DEBUG, f"Writing b'AT\\r' to port {self._port}")
    self._ser.write(b'AT\r')

    t = time()
    ret = ''
    while time() - t < 2:
      # Reading the answers
      if ret := self._ser.readline().strip().decode():
        self.log(logging.DEBUG, f"Read {ret} from port {self._port}")
      # Just waiting for the answer to be 'OK'
      if 'OK' in ret:
        self.log(logging.INFO, "Successfully connected to Sim868")
        break
      sleep(0.1)

    # Raising if no response after 2 seconds
    if 'ERROR' in ret:
      raise IOError("Got an ERROR message from the Sim868")
    elif 'OK' not in ret:
      raise ConnectionError("Could not get answer from Sim868 after 2 seconds")

    # Checking if the status of the SIM card is OK
    self._ser.write(b'AT+CPIN?\r')
    self.log(logging.DEBUG, f"Writing b'AT+CPIN?\\r' to port {self._port}")

    # Waiting for a response from the Sim868
    t = time()
    ret = ''
    need_pin = False
    while time() - t < 2:
      # Reading the answers
      if ret := self._ser.readline().strip().decode():
        self.log(logging.DEBUG, f"Read {ret} from port {self._port}")

      # Expecting an answer giving the SIM status, and parsing it
      if (status := fullmatch(r'\+CPIN:\s(.+)', ret)) is not None:
        status, = status.groups()
        # Several possible answers from the Sim868
        if status == 'READY':
          self.log(logging.INFO, "SIM card not requiring a pin code")
        elif status == 'SIM PIN':
          self.log(logging.INFO, "SIM card requiring a pin code")
          need_pin = True
        else:
          raise IOError(f"Got CPIN status from the Sim868: {status}, but the "
                        f"InOut does not implement this case")

      # Only exiting when receiving the final answer 'OK'
      if 'OK' in ret:
        self.log(logging.INFO, "Successfully read CPIN from Sim868")
        break
      sleep(0.1)

    # Raising if no response after 2 seconds
    if 'ERROR' in ret:
      raise IOError("Got an ERROR message from the Sim868")
    elif 'OK' not in ret:
      raise ConnectionError("Could not get answer from Sim868 after 2 seconds")

    # Case when a PIN code is needed
    if need_pin:
      # No PIN code was given
      if self._pin is None:
        raise ValueError("A PIN code is needed to activate the SIM card but "
                         "none was given")
      else:
        # Setting the PIN code for the SIM card
        self._ser.write(f'AT+CPIN={self._pin}\r'.encode())
        self.log(logging.DEBUG, f"Writing b'AT+CPIN={self._pin}\\r' to port "
                                f"{self._port}")

        # Waiting for a response from the Sim868
        t = time()
        ret = ''
        while time() - t < 2:
          # Reading the answers
          if ret := self._ser.readline().strip().decode():
            self.log(logging.DEBUG, f"Read {ret} from port {self._port}")

          # Only exiting when receiving the final answer 'OK'
          if 'OK' in ret:
            self.log(logging.INFO, "Successfully sent the PIN code")
            break
          sleep(0.1)

        # Raising if no response after 2 seconds
        if 'ERROR' in ret:
          raise IOError("Got an ERROR message from the Sim868")
        elif 'OK' not in ret:
          raise ConnectionError(
              "Could not get answer from Sim868 after 2 seconds")

        # Giving some time to the Sim868 for setting the PIN code
        sleep(1)

        # Checking again if the status of the SIM card is OK
        self._ser.write(b'AT+CPIN?\r')
        self.log(logging.DEBUG, f"Writing b'AT+CPIN?\\r' to port {self._port}")

        # Waiting for a response from the Sim868
        t = time()
        ret = ''
        while time() - t < 2:
          # Reading the answers
          if ret := self._ser.readline().strip().decode():
            self.log(logging.DEBUG, f"Read {ret} from port {self._port}")

          # Expecting an answer giving the SIM status, and parsing it
          if (status := fullmatch(r'\+CPIN:\s(.+)', ret)) is not None:
            status, = status.groups()
            # Several possible answers from the Sim868
            if status == 'READY':
              self.log(logging.INFO, "Successfully set the PIN code")
            else:
              raise IOError("PIN still not ready after setting the PIN code")

          # Only exiting when receiving the final answer 'OK'
          if 'OK' in ret:
            self.log(logging.INFO, "Successfully read CPIN from Sim868")
            break
          sleep(0.1)

        # Raising if no response after 2 seconds
        if 'ERROR' in ret:
          raise IOError("Got an ERROR message from the Sim868")
        elif 'OK' not in ret:
          raise ConnectionError(
              "Could not get answer from Sim868 after 2 seconds")

    # Checking that the SIM card is registered with an operator
    registered = False
    t = time()
    while time() - t < self._reg_timeout:

      # Requesting the registration status from the Sim868
      self._ser.write(b'AT+CGREG?\r')
      self.log(logging.DEBUG, f"Writing b'AT+CGREG?\\r' to port {self._port}")

      ret = ''
      t1 = time()
      # Reading the answers
      while time() - t < self._reg_timeout:
        # Sometimes the Sim868 sends back non UTF-8 characters, ignoring them
        if ret := self._ser.readline().strip().decode():
          self.log(logging.DEBUG, f"Read {ret} from port {self._port}")

        # Expecting an answer giving the registration status, and parsing it
        if (status := fullmatch(r'\+CGREG:\s\d,(\d).*', ret)) is not None:
          status, = status.groups()
          # Several possible answers from the Sim868
          if status in ('0', '3', '4'):
            raise IOError(f"The Sim868 is not connected to a network, and is "
                          f"not searching anymore")
          elif status == '2':
            self.log(logging.INFO, "The Sim868 is searching for an operator "
                                   "to register with")
          elif status in ('1', '5'):
            self.log(logging.INFO, "Sim868 successfully registered with an "
                                   "operator")
            registered = True

        # Only exiting when receiving the final answer 'OK'
        if 'OK' in ret:
          self.log(logging.DEBUG, "Successfully read registration status from "
                                  "Sim868")
          break
        sleep(0.1)

      # Raising if no response after the given timeout
      if 'ERROR' in ret:
        raise IOError("Got an ERROR message from the Sim868")
      elif 'OK' not in ret:
        raise ConnectionError(f"Could not get answer from Sim868 after "
                              f"{t + self._reg_timeout - t1} seconds")

      # Exiting the loop once the Sim868 is connected to a network
      if registered:
        self.log(logging.INFO, "Sim868 successfully registered with an "
                               "operator")
        break
      sleep(self._reg_timeout / 5)

    if not registered:
      raise ConnectionError("The Sim868 was not able to register with an "
                            "operator before the given timeout")

    # Setting the message input mode to Text
    self._ser.write(b'AT+CMGF=1\r')
    self.log(logging.DEBUG, f"Writing b'AT+CMGF=1\\r' to port {self._port}")

    # Waiting for a response from the Sim868
    t = time()
    ret = ''
    while time() - t < 9:
      # Reading the answers
      if ret := self._ser.readline().strip().decode():
        self.log(logging.DEBUG, f"Read {ret} from port {self._port}")

      # Only exiting when receiving the final answer 'OK'
      if 'OK' in ret:
        self.log(logging.INFO, "Successfully set the message input mode")
        break
      sleep(0.1)

    # Raising if no response after 9 seconds
    if 'ERROR' in ret:
      raise IOError("Got an ERROR message from the Sim868")
    elif 'OK' not in ret:
      raise ConnectionError("Could not get answer from Sim868 after 9 seconds")

  def set_cmd(self, *cmd: str) -> None:
    """Sends an SMS whose text is the :obj:`str` received as command to all the
    phone numbers.

    If multiple messages are received, sends them all in the received order. If
    the messages are not :obj:`str`, they are first converted to strings if
    possible.
    """

    # Converting the messages to string
    cmd = map(str, cmd)

    # Iterating over all the messages to send
    for msg in cmd:
      # Not sending if the message is empty
      if msg:
        # Iterating over all the destination numbers
        for nr in self._numbers:

          # Providing the number to send the message to
          self._ser.write(f'AT+CMGS="{nr}"\r'.encode())
          self.log(logging.DEBUG, f"Writing b'AT+CMGS=\"{nr}\"\\r' to port "
                                  f"{self._port}")

          # Waiting for a response from the Sim868
          t = time()
          ret = ''
          while time() - t < 2:
            # Reading the answers
            while self._ser.inWaiting() > 0:
              ret += self._ser.read().decode()
            if ret:
              self.log(logging.DEBUG, f"Read {ret} from port {self._port}")

            # Only exiting when receiving the > character
            if '> ' in ret:
              self.log(logging.INFO, "Received stop character for providing "
                                     "the message")
              break
            sleep(0.1)

          # Raising if no stop character after 2 seconds
          if 'ERROR' in ret:
            raise IOError("Got an ERROR message from the Sim868")
          elif '> ' not in ret:
            raise ValueError("Did not receive the stop character in 2 seconds")

          # Providing the text message to send
          self._ser.write(f"{msg}\x1a".encode())
          self.log(logging.DEBUG, f"Writing b'{msg}\\x1a' to port "
                                  f"{self._port}")

          # Waiting for a response from the Sim868
          t = time()
          ret = ''
          while time() - t < 2:
            # Reading the answers
            if ret := self._ser.readline().strip().decode():
              self.log(logging.DEBUG, f"Read {ret} from port {self._port}")

            # Only exiting when receiving the final answer 'OK'
            if 'OK' in ret:
              self.log(logging.INFO, "Successfully sent the message")
              break
            sleep(0.1)

          # Raising if no response after 2 seconds
          if 'ERROR' in ret:
            raise IOError("Got an ERROR message from the Sim868")
          elif 'OK' not in ret:
            raise ConnectionError(
              "Could not get answer from Sim868 after 2 seconds")

  def close(self) -> None:
    """Closes the serial port."""

    if self._ser is not None:
      self.log(logging.INFO, f"Closing the serial port {self._port}")
      self._ser.close()
