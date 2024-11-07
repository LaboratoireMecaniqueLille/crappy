# coding: utf-8

from struct import unpack
from time import time
from typing import Optional, Union
from collections.abc import Callable, Iterable
import logging

from .meta_block import Block
from .._global import OptionalModule

try:
  from serial import Serial
  from serial.serialutil import SerialException
except (ModuleNotFoundError, ImportError):
  Serial = OptionalModule("pyserial")
  SerialException = OptionalModule("pyserial")


class UController(Block):
  """Block for interfacing over serial with an external device, written mostly
  for communication with microcontrollers.

  It can send labeled commands to the device, and/or receive labeled data from
  it. This Block is meant to be used along with the `microcontroller.py`
  MicroPython template located in the `tool` folder of Crappy, even though it
  is not mandatory. A given syntax needs to be followed for any data to be
  exchanged.
  
  .. versionadded:: 1.5.8
  """

  def __init__(self,
               labels: Optional[Union[str, Iterable[str]]] = None,
               cmd_labels: Optional[Union[str, Iterable[str]]] = None,
               init_output: Optional[dict[str, float]] = None,
               post_process: Optional[dict[str,
                                           Callable[[float], float]]] = None,
               t_device: bool = False,
               port: str = '/dev/ttyUSB0',
               baudrate: int = 115200,
               display_freq: bool = False,
               freq: Optional[float] = 100,
               debug: Optional[bool] = False) -> None:
    """Checks the validity of the arguments.

    Args:
      labels: An iterable (like a :obj:`list` or a :obj:`tuple`) containing the
        labels to get from the device (as :obj:`str`). Only these labels should
        be given as argument to the
        :meth:`~crappy.blocks.UController.send_to_pc` method in the MicroPython
        script. If this argument is not :obj:`None`, then the ``init_output``
        argument should be given as well. No more than 9 labels should be
        given. If there's only one label to acquire, it can be given directly
        as a :obj:`str` and not in an iterable.
      cmd_labels: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        the command labels that will be sent to the device upon reception from
        an upstream Block. The variables in the MicroPython script should have
        these exact names. Not more than 9 cmd_labels should be given. If
        there's only one command label, it can be given directly as a
        :obj:`str` and not in an iterable.
      init_output: If the ``labels`` argument is not :obj:`None`, the values to 
        output to downstream Blocks for each label as long as no value has been 
        received from the device. An initial output value must be given for 
        each label.
      post_process: Optionally allows applying a function to the data of a 
        label before transmitting it to downstream Blocks. It is possible to 
        give functions for only part of the labels.
      t_device: It :obj:`True`, the timestamp returned under the label `'t(s)'` 
        is the one of the device, not the one of Crappy. It may reduce the 
        maximum achievable sample rate, as more bytes have to be transmitted, 
        but it is also far more precise.
      port: The serial port to open for communicating with the device. In 
        Windows, they are usually called `COMx`, whereas in Linux and Mac 
        they're called `/dev/ttyxxxx`.
      baudrate: The baudrate for serial communication. It depends on the 
        capabilities of the device.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
        
        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      freq: The target looping frequency for the Block. If :obj:`None`, loops 
        as fast as possible.
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
        
        .. versionadded:: 2.0.0
    """

    self._bus = None

    super().__init__()
    self.debug = debug

    if not isinstance(display_freq, bool):
      raise TypeError("display_freq should be either True or False !")
    self.display_freq = display_freq

    if not isinstance(freq, float) and not isinstance(freq, int) or freq <= 0:
      raise TypeError("freq should be a positive float !")
    self.freq = freq

    if not isinstance(t_device, bool):
      raise TypeError("t_device should be either True or False !")
    self._t_device = t_device

    if not isinstance(port, str):
      raise TypeError("port should be a string !")
    self._port = port

    if not isinstance(baudrate, int) or baudrate < 0:
      raise ValueError("baudrate should be a positive integer !")
    self._baudrate = baudrate

    # Forcing the labels into a list
    if labels is not None and isinstance(labels, str):
      self._labels = [labels]
    elif labels is not None:
      self._labels = list(labels)
    else:
      self._labels = None

    # Forcing the cmd_labels into a list
    if cmd_labels is not None and isinstance(cmd_labels, str):
      self._cmd_labels = [cmd_labels]
    elif cmd_labels is not None:
      self._cmd_labels = list(cmd_labels)
    else:
      self._cmd_labels = None

    if self._labels is not None and len(self._labels) > 9:
      raise ValueError("Sorry, a maximum of 9 labels is allowed !")
    if self._cmd_labels is not None and len(self._cmd_labels) > 9:
      raise ValueError("Sorry, a maximum of 9 cmd_labels is allowed !")

    if self._cmd_labels is not None:
      self._prev_cmd = {cmd_label: None for cmd_label in self._cmd_labels}

    if init_output is not None and not isinstance(init_output, dict):
      raise TypeError("init_output should be a dict !")
    if self._labels is not None and not all(
        label in (init_output if init_output is not None else dict())
        for label in self._labels):
      raise ValueError("Every label should have an init_output value !")
    self._out = init_output

    if post_process is not None and (not isinstance(post_process, dict) or
                                     not all(callable(func) for func
                                             in post_process.values())):
      raise TypeError("post_process should be a dict of callables !")
    self._post_process = post_process if post_process is not None else {}

    self._buffer = None
    self._cmd_table = None
    self._labels_table = None

  def prepare(self) -> None:
    """Opens the serial port, and sends a `'go'` message to the device.

    Also shares with the device two tables associating each `cmd_label` and
    `label` with an integer. This allows reducing the traffic on the serial
    bus.

    Note:
      The commands are sent as text because some boards cannot read bytes from
      the `stdin` buffer in MicroPython.
    """

    # Checking if the link layout is relevant with respect to the arguments
    if self._labels is not None and not self.outputs:
      raise IOError("labels are specified but there's no output link !")
    if self._cmd_labels is not None and not self.inputs:
      raise IOError("cmd_labels are specified but there's no input link !")

    # Buffer for storing the received bytes
    if self._labels is not None:
      self._buffer = b''

    # Opening the serial port
    try:
      self.log(logging.INFO, f"Opening the serial port {self._port} with "
                             f"baudrate {self._baudrate}")
      self._bus = Serial(self._port,
                         self._baudrate,
                         timeout=0,
                         write_timeout=0)
    except SerialException:
      raise IOError(f"Couldn't connect to the device on the port {self._port}")

    # Assigning indexes to the cmd_labels and labels, to identify them easily
    # and reduce the traffic on the bus
    if self._cmd_labels is not None:
      self._cmd_table = {label: i for i, label
                         in enumerate(self._cmd_labels, start=1)}
    else:
      self._cmd_table = dict()
    self.log(logging.DEBUG, f"Command table : {self._cmd_table}")

    if self._labels is not None:
      self._labels_table = {label: i for i, label
                            in enumerate(self._labels, start=1)}
    else:
      self._labels_table = dict()

    # The presence of the label 't(s)' indicates that the device should return
    # a timestamp along with the data
    if self._labels is not None and self._t_device:
      self._labels_table |= {'t(s)': 0}

    self.log(logging.DEBUG, f"Labels table : {self._labels_table}")

    # Emptying the read buffer before starting
    try:
      self._bus.reset_input_buffer()
      self._bus.reset_output_buffer()
    except SerialException:
      raise IOError(f"Reading from the device on port {self._port} failed, "
                    f"it may have been disconnected.")

    # Sending the 'go' command to start the device
    self.log(logging.INFO, f"Sending start command on port {self._port}")
    try:
      msg = b''.join((b'go', str(len(self._cmd_table)).encode(),
                      str(len(self._labels_table)).encode(), b'\r\n'))
      self._bus.write(msg)
      self.log(logging.DEBUG, f"Sent {msg} on the port {self._port}")
    except SerialException:
      raise IOError(f"Writing to the device on port {self._port} failed, "
                    f"it may have been disconnected.")

    # Sending the table of cmd_labels and their indexes
    self.log(logging.INFO, f"Sending the command labels table on port "
                           f"{self._port}")
    for cmd, i in self._cmd_table.items():
      try:
        msg = b''.join((str(i).encode(), cmd.encode(), b'\r\n'))
        self._bus.write(msg)
        self.log(logging.DEBUG, f"Sent {msg} on the port {self._port}")
      except SerialException:
        raise IOError(f"Writing to the device on port {self._port} failed, "
                      f"it may have been disconnected.")

    # Sending the table of labels and their indexes
    self.log(logging.INFO, f"Sending the labels table on port {self._port}")
    for label, i in self._labels_table.items():
      try:
        msg = b''.join((str(i).encode(), label.encode(), b'\r\n'))
        self._bus.write(msg)
        self.log(logging.DEBUG, f"Sent {msg} on the port {self._port}")
      except SerialException:
        raise IOError(f"Writing to the device on port {self._port} failed, "
                      f"it may have been disconnected.")

  def loop(self) -> None:
    """First sends the commands from upstream Blocks to the device, then reads
    the data from the device and sends it to the downstream Blocks.

    Important:
      The precision of the commands sent to the device is limited to 3 digits
      after the decimal point, to limit the traffic on the bus. Adapt the range
      of the command values consequently.

    Note:
      Commands are sent as text, because some boards cannot read bytes from the
      `stdin` buffer in MicroPython. Data is however received on the PC from
      the device as bytes.
    """

    """Loop for sending the commands to the device.
    
    First, for each label in the cmd_labels list we search for a matching 
    command in the upstream links. Only the first one found is considered. If 
    the command value is different from the last value of this label, then the 
    command is sent to the device. Otherwise it is ignored. The command value
    is then stored as the last value of the label.
    """
    if self._cmd_labels is not None:
      cmd = self.recv_last_data()
      for label in self._cmd_labels:
        if label in cmd and cmd[label] != self._prev_cmd[label]:
          msg = b''.join((f'{self._cmd_table[label]}'
                          f'{cmd[label]:.3f}'.encode(),
                          b'\r\n'))

          # Information for debugging
          self.log(logging.DEBUG, f"Sending {msg} on the port {self._port}")

          # Sending the actual message to the device
          try:
            self._bus.write(msg)
          except SerialException:
            raise IOError(f"Writing to the device on port {self._port} "
                          f"failed, it may have been disconnected.")
          self._prev_cmd[label] = cmd[label]

    """Loop for receiving data from the device.
    
    A given number of bytes is read from the device, depending if `t_device` is
    True or False. they are then parsed to extract the index of the label, the
    data, and optionally the timestamp. If a label with a matching index is
    present in labels, its value is updated, as well as the timestamp. The
    values of ALL the labels are then sent to downstream blocks.
    """
    if self._labels is not None:
      # Reading the message from the device
      retries = 3
      while retries:
        try:
          recv = self._bus.read()
        except SerialException:
          raise IOError(f"Reading from the device on port {self._port} "
                        f"failed, it may have been disconnected.")

        # This prevents the loop from staying stuck with an incomplete reading
        if not recv:
          retries -= 1
          continue

        self._buffer += recv
        # Exiting the loop when the desired number of bytes is reached
        if (self._t_device and len(self._buffer) == 9) or \
            (not self._t_device and len(self._buffer) == 5):
          break

      # There was no message from the device, or it was incomplete
      if (self._t_device and len(self._buffer) != 9) or \
          (not self._t_device and len(self._buffer) != 5):
        return

      # Information for debugging
      self.log(logging.DEBUG, f"Received {self._buffer} on the port "
                              f"{self._port}")

      # Parsing the received bytes
      read = unpack('<ibf' if self._t_device else '<bf', self._buffer)
      self._buffer = b''

      # Reading the timestamp if relevant
      if self._t_device:
        self._out['t(s)'] = read[0] / 1000
        read = read[1:]

      self.log(logging.DEBUG, f"Read value {read} from the device")

      # Updating the label value and sending to the downstream blocks
      for label in self._labels:
        if read[0] == self._labels_table[label]:
          value = read[1]
          self._out[label] = self._post_process[label](value) if \
              label in self._post_process else value
          if not self._t_device:
            self._out['t(s)'] = time() - self.t0
          self.send(self._out)
          break

    # Emptying the port buffer even if the messages are not processed
    else:
      try:
        self._bus.reset_input_buffer()
      except SerialException:
        raise IOError(f"Reading from the device on port {self._port} "
                      f"failed, it may have been disconnected.")

  def finish(self) -> None:
    """Closes the serial port, and sends a `'stop!'` message to the device."""

    if self._bus is not None:
      # Sending a 'stop!' message to the device
      self.log(logging.INFO, f"Sending stop command on port {self._port}")
      try:
        msg = b'stop!\r\n'
        self._bus.write(msg)
        self.log(logging.DEBUG, f"Sent {msg} on the port {self._port}")
      except SerialException:
        pass

      self.log(logging.INFO, "Closing the serial connection")
      self._bus.close()
