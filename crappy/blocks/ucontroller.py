# coding: utf-8

from .block import Block
from .._global import OptionalModule
from struct import unpack
from time import time
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
  it. This block is meant to be used along with the MicroController.py
  MicroPython template located in the `tool` folder of Crappy, even though it
  is not mandatory. A given syntax needs to be followed for any data to be
  exchanged.
  """

  def __init__(self,
               labels: list = None,
               cmd_labels: list = None,
               init_output: dict = None,
               post_process: dict = None,
               t_device: bool = False,
               port: str = '/dev/ttyUSB0',
               baudrate: int = 115200,
               verbose: bool = False,
               freq: float = 100) -> None:
    """Checks the validity of the arguments.

    Args:
      labels (:obj:`list`, optional): The list of the labels to get from the
        device. Only these labels should be given as argument to the
        :meth:`send_to_pc` method in the MicroPython script. If this argument
        is not :obj:`None`, then the``init_output`` argument should be given as
        well. No more than 9 labels should be given.
      cmd_labels (:obj:`list`, optional): The list of the command labels that
        will be sent to the device upon reception from an upstream block. The
        variables in the MicroPython script should have these exact names. No
        more than 9 cmd_labels should be given.
      init_output (:obj:`dict`, optional): If the ``labels`` argument is not
        :obj:`None`, the values to output to downstream blocks for each label
        as long as no value has been received from the device. An initial
        output value must be given for each label.
      post_process (:obj:`dict`, optional): Optionally allows applying a
        function to the data of a label before transmitting it to downstream
        blocks. It is possible to give functions for only part of the labels.
      t_device (:obj:`bool`, optional): It :obj:`True`, the timestamp returned
        under the label `'t(s)'` is the one of the device, not the one of
        Crappy. It may reduce the maximum achievable sample rate, as more bytes
        have to be transmitted, but it is also far more precise.
      port (:obj:`str`, optional): The serial port to open for communicating
        with the device. In Windows, they are usually called `COMx`, whereas in
        Linux and Mac they're called `/dev/ttyxxxx`.
      baudrate (:obj:`int`, optional): The baudrate for serial communication.
        It depends on the capabilities of the device.
      verbose (:obj:`bool`, optional): If :obj:`True`, prints debugging
        information.
      freq (:obj:`float`, optional): The looping frequency of the block.
    """

    super().__init__()

    if not isinstance(verbose, bool):
      raise TypeError("verbose should be either True or False !")
    self.verbose = verbose

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

    if labels is not None and not isinstance(labels, list):
      raise TypeError('labels should be a list !')
    if labels is not None and len(labels) > 9:
      raise ValueError("Sorry, a maximum of 9 labels is allowed !")
    if cmd_labels is not None and not isinstance(cmd_labels, list):
      raise TypeError('cmd_labels should be a list !')
    if cmd_labels is not None and len(cmd_labels) > 9:
      raise ValueError("Sorry, a maximum of 9 cmd_labels is allowed !")
    self._labels = labels
    self._cmd_labels = cmd_labels

    if self._cmd_labels is not None:
      self._prev_cmd = {cmd_label: None for cmd_label in self._cmd_labels}

    if init_output is not None and not isinstance(init_output, dict):
      raise TypeError("init_output should be a dict !")
    if labels is not None and not all(label in (init_output if init_output
                                                is not None else {})
                                      for label in labels):
      raise ValueError("Every label should have an init_output value !")
    self._out = init_output

    if post_process is not None and (not isinstance(post_process, dict) or
                                     not all(callable(func) for func
                                             in post_process.values())):
      raise TypeError("post_process should be a dict of callables !")
    self._post_process = post_process if post_process is not None else {}

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
      self._bus = Serial(self._port,
                         self._baudrate,
                         timeout=0,
                         write_timeout=0)
    except SerialException:
      raise IOError("Couldn't connect to the device on the port "
                    "{}".format(self._port))

    # Assigning indexes to the cmd_labels and labels, to identify them easily
    # and reduce the traffic on the bus
    self._cmd_table = {label: i for i, label in enumerate(self._cmd_labels,
                                                          start=1)} \
        if self._cmd_labels is not None else {}
    self._labels_table = {label: i for i, label in enumerate(self._labels,
                                                             start=1)} \
        if self._labels is not None else {}
    # The presence of the label 't(s)' indicates that the device should return
    # a timestamp along with the data
    if self._labels is not None and self._t_device:
      self._labels_table.update({'t(s)': 0})

    # Emptying the read buffer before starting
    while True:
      try:
        recv = self._bus.read()
      except SerialException:
        raise IOError("Reading from the device on port {} failed, it may"
                      "have been disconnected.".format(self._port))

      if not recv:
        break

    # Sending the 'go' command to start the device
    try:
      self._bus.write(b'go' +
                      str(len(self._cmd_table)).encode() +
                      str(len(self._labels_table)).encode() +
                      b'\r\n')
      if self.verbose:
        print('[UController] Sent {} on the port '
              '{}'.format(b'go' + str(len(self._cmd_table)).encode() +
                          str(len(self._labels_table)).encode() +
                          b'\r\n', self._port))
    except SerialException:
      raise IOError("Writing to the device on port {} failed, it may"
                    "have been disconnected.".format(self._port))

    # Sending the table of cmd_labels and their indexes
    for cmd, i in self._cmd_table.items():
      try:
        self._bus.write(str(i).encode() + cmd.encode() + b'\r\n')
        if self.verbose:
          print('[UController] Sent {} on the port {}'.format(
              str(i).encode() + cmd.encode() + b'\r\n', self._port))
      except SerialException:
        raise IOError("Writing to the device on port {} failed, it may"
                      "have been disconnected.".format(self._port))

    # Sending the table of labels and their indexes
    for label, i in self._labels_table.items():
      try:
        self._bus.write(str(i).encode() + label.encode() + b'\r\n')
        if self.verbose:
          print('[UController] Sent {} on the port {}'.format(
            str(i).encode() + label.encode() + b'\r\n', self._port))
      except SerialException:
        raise IOError("Writing to the device on port {} failed, it may"
                      "have been disconnected.".format(self._port))

  def loop(self) -> None:
    """First sends the commands from upstream blocks to the device, then reads
    the data from the device and sends it to the downstream blocks.

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
      received_cmd = [link.recv_chunk() if link.poll() else {} for link
                      in self.inputs]
      for label in self._cmd_labels:
        for dic in received_cmd:
          if label in dic:
            for value in dic[label]:
              if value != self._prev_cmd[label]:

                # Information for debugging
                if self.verbose:
                  print("[UController] Sent {} on the "
                        "port {}".format((str(self._cmd_table[label]) +
                                          str('%.3f' % value)).encode() +
                                         b'\r\n',
                                         self._port))

                # Sending the actual message to the device
                try:
                  self._bus.write((str(self._cmd_table[label]) +
                                   str('%.3f' % value)).encode() +
                                  b'\r\n')
                except SerialException:
                  raise IOError("Writing to the device on port {} failed, it "
                                "may have been "
                                "disconnected.".format(self._port))
                self._prev_cmd[label] = value
            break

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
      while True:
        try:
          recv = self._bus.read()
        except SerialException:
          raise IOError("Reading from the device on port {} failed, it may"
                        "have been disconnected.".format(self._port))

        # This prevents the loop from staying stuck with an incomplete reading
        if not recv:
          retries -= 1
          if not retries:
           break
          continue

        self._buffer += recv
        # Exiting the loop when the desired number of bytes is reached
        if (self._t_device and
            len(self._buffer) == 9) or (not self._t_device and
                                        len(self._buffer) == 5):
          break

      # There was no message from the device, or it was incomplete
      if (self._t_device and
          len(self._buffer) != 9) or (not self._t_device and
                                      len(self._buffer) != 5):
        return

      # Information for debugging
      if self.verbose:
        print("[UController] Received {} on the port {}".format(self._buffer,
                                                                self._port))

      # Parsing the received bytes
      read = unpack('<ibf' if self._t_device else '<bf', self._buffer)
      self._buffer = b''

      # Reading the timestamp if relevant
      if self._t_device:
        self._out['t(s)'] = read[0] / 1000
        read = read[1:]

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
      while True:
        try:
          recv = self._bus.read()
        except SerialException:
          raise IOError("Reading from the device on port {} failed, it may"
                        "have been disconnected.".format(self._port))

        if not recv:
          break

        if self.verbose:
          print("[UController] Received {} on the port {}".format(recv,
                                                                  self._port))

  def finish(self) -> None:
    """Closes the serial port, and sends a `'stop!'` message to the device."""

    # Sending a 'stop!' message to the device
    try:
      self._bus.write(b'stop!' + b'\r\n')
      if self.verbose:
        print("[UController] Sent {} on the port {}".format(b'stop!' + b'\r\n',
                                                            self._port))
    except SerialException:
      pass

    self._bus.close()
