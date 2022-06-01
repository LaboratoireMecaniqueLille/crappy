# coding: utf-8

from typing import Union, List, Optional
from .block import Block
from ..inout import inout_dict


class IOBlock(Block):
  """This block is meant to drive :ref:`In / Out` objects. It can acquire data,
  and/or set commands. One IOBlock can only drive a single InOut.

  If it has incoming links, it will set the commands received over the labels
  given in ``cmd_labels``. Additional commands to set at the very beginning or
  the very end of the test can also be specified.

  If it has outgoing links, it will acquire data and send it downstream over
  the labels given in ``labels``. It is possible to trigger the acquisition
  using a predefined label.


  """

  def __init__(self,
               name: str,
               labels: Optional[List[str]] = None,
               cmd_labels: Optional[List[str]] = None,
               trigger_label: Optional[str] = None,
               streamer: bool = False,
               initial_cmd: Optional[Union[list]] = None,
               exit_cmd: Optional[list] = None,
               make_zero_delay: Optional[float] = None,
               spam: bool = False,
               freq: Optional[float] = None,
               verbose: bool = False,
               **kwargs) -> None:
    """Sets the args and initializes the parent class.

    Args:
      name: The name of the :ref:`In / Out` class to instantiate.
      labels: A :obj:`list` containing the output labels for InOuts that
        acquire data. They correspond to the values returned by the InOut's
        :meth:`get_data` method, so there should be as many labels as values
        returned, and given in the appropriate order. The first label must
        always be the timestamp, preferably called ``'t(s)'``. This argument
        can be omitted if :meth:`get_data` returns a :obj:`dict` instead of a
        :obj:`list`. Ignored if the block has no output link.
      cmd_labels: A :obj:`list` of the labels considered as inputs for this
        block, for InOuts that set commands. The values received from these
        labels will be passed to the InOut's :meth:`set_cmd` method, in the
        same order as the labels are given. Usually, time is not part of the
        cmd_labels. Ignored if the block has no input link.
      trigger_label: If given, the block will only read data whenever a value
        (can be any value) is received on this label. Ignored if the block has
        no output link. A trigger label can also be a cmd label.
      streamer: If :obj:`False`, the :meth:`get_data` method of the InOut
        object is called for acquiring data, else it's the :meth:`get_stream`
        method.
      initial_cmd: An initial command for the InOut, set during
        :meth:`prepare`. If given, there must be as many values as in
        cmd_labels.
      exit_cmd: A final command for the InOut, set during :obj:`finish`. If
        given, there must be as many values as in cmd_labels.
      make_zero_delay: If set, will acquire data before the beginning of the
        test and use it to offset all the labels to zero. The data will be
        acquired during the given number of seconds. Ignored if the block has
        no output links.
      spam: If :obj:`False`, the block will call :meth:`set_cmd` on the
        InOut object only if the current command is different from the
        previous.
      freq: The block will try to loop as this frequency, or as fast as
        possible if no value is given.
      verbose: If :obj:`True`, prints the looping frequency of the block.
      **kwargs: The arguments to be passed to the :ref:`In / Out` class.
    """

    super().__init__()
    self.niceness = -10
    if freq is not None:
      self.freq = freq
    self.verbose = verbose

    # The label argument can be omitted for streaming
    if labels is None and streamer:
      self.labels = ['t(s)', 'stream']
    else:
      self.labels = labels

    # The labels to get
    self._cmd_labels = cmd_labels
    self._trig_label = trigger_label

    # Checking that the initial_cmd and exit_cmd length are consistent
    if cmd_labels is not None:
      if initial_cmd is not None and len(initial_cmd) != len(cmd_labels):
        raise ValueError("There should be as many values in initial_cmd as "
                         "there are in cmd_labels !")
      if exit_cmd is not None and len(exit_cmd) != len(cmd_labels):
        raise ValueError("There should be as many values in exit_cmd as "
                         "there are in cmd_labels !")

    self._initial_cmd = initial_cmd
    self._exit_cmd = exit_cmd

    self._streamer = streamer
    self._spam = spam
    self._make_zero_delay = make_zero_delay

    self._stream_started = False
    self._last_cmd = None
    self._prev_value = dict()

    # Instantiating the device
    self._device = inout_dict[name.capitalize()](**kwargs)

  def prepare(self) -> None:
    """Checks the consistency of the link layout, opens the device and sets the
    initial command if required."""

    # Checking that the block has inputs or outputs
    if not self.inputs and not self.outputs:
      raise IOError('Error ! The IOBlock is neither an input nor an output !')

    # cmd_labels must be defined when the block has inputs
    if self.inputs and self._cmd_labels is None:
      raise ValueError('Error ! The IOBlock has incoming links but no '
                       'cmd_labels have been given !')

    self._read = bool(self.outputs)
    self._write = bool(self.inputs)

    # Now opening the device
    self._device.open()

    # Acquiring data for offsetting the output
    if self._read and self._make_zero_delay is not None:
      self._device.make_zero(self._make_zero_delay)

    # Writing the first command before the beginning of the test if required
    if self._write and self._initial_cmd is not None:
      self._device.set_cmd(*self._initial_cmd)
      self._last_cmd = self._initial_cmd

  def loop(self) -> None:
    """Gets the latest command, reads data from the device and sets the
    command.

    Also handles the trig label if one was given, and manages the buffer for
    the previously received commands.
    """

    # Receiving all the latest data waiting in the links
    # Cannot use self.get_last because trig_label needs a special handling
    recv_data = dict()
    for link in self.inputs:
      latest = link.recv_last(blocking=False)
      if latest is not None:
        recv_data.update(latest)

    # Reading data from the device if there's no trig_label or if data has been
    # received on this trig_label
    if self._read:
      if self._trig_label is None or self._trig_label in recv_data:
        self._read_data()

    # Storing the latest received values in the buffer
    self._prev_value.update(recv_data)
    # Completing the values that may be missing in recv_data
    recv_data.update(self._prev_value)

    if self._write:
      # At the very beginning of the test, there may not be any received value
      if not recv_data:
        return

      # Keeping only the labels in cmd_labels
      cmd = [val for label, val in recv_data.items()
             if label in self._cmd_labels]

      # If not all cmd_labels have a value, returning without calling set_cmd
      if len(cmd) != len(self._cmd_labels):
        print(f"WARNING ! Not enough values received in the "
              f"{type(self._device).__name__} InOut to set the cmd, cmd "
              f"not set !")
        return

      # Setting the command if it's different from the previous or spam is True
      if cmd != self._last_cmd or self._spam:
        self._device.set_cmd(*cmd)
        self._last_cmd = cmd

  def finish(self) -> None:
    """Stops the stream, sets the exit command if necessary, and closes the
    device."""

    # Stopping the stream
    if self._streamer:
      self._device.stop_stream()

    # Setting the exit command
    if self._write and self._exit_cmd is not None:
      self._device.set_cmd(*self._exit_cmd)

    # Closing the device
    self._device.close()

  def _read_data(self) -> None:
    """Reads the data or the stream, offsets the timestamp and sends the data
    to downstream blocks."""

    if self._streamer:
      # Starting the stream if needed
      if not self._stream_started:
        self._device.start_stream()
        self._stream_started = True
      # Actually getting the stream
      data = self._device.get_stream()
    else:
      # Regular reading of data
      data = self._device.return_data()

    if data is None:
      return

    # Making time relative to the beginning of the test
    if isinstance(data, dict) and 't(s)' in data:
      data['t(s)'] -= self.t0
    else:
      data[0] -= self.t0

    self.send(data)
