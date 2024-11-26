# coding: utf-8

from typing import Union, Optional, Any
from collections.abc import Iterable
import logging

from .meta_block import Block
from ..inout import inout_dict, InOut, deprecated_inouts
from ..tool.ft232h import USBServer


class IOBlock(Block):
  """This Block is meant to drive :class:`~crappy.inout.InOut` objects. It can
  acquire data, and/or set commands. One IOBlock can only drive a single InOut.

  If it has incoming :class:`~crappy.links.Link`, it will set the commands 
  received over the labels given in ``cmd_labels`` by calling the 
  :meth:`~crappy.inout.InOut.set_cmd` method of the InOut. Additional commands 
  to set at the very beginning or the very end of the test can also be 
  specified.

  If it has outgoing :class:`~crappy.links.Link`, it will acquire data using 
  the :meth:`~crappy.inout.InOut.get_data` method of the InOut and send it
  downstream over the labels given in ``labels``. It is possible to trigger the
  acquisition using a predefined label.

  The ``streamer`` argument allows using the "streamer" mode of InOuts
  supporting it, instead of the regular acquisition mode. Finally, the
  ``make_zero_delay`` argument allows offsetting the acquired values to zero at
  the beginning of the test. Refer to the documentation of each argument for a
  more detailed description.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               name: str,
               labels: Optional[Union[str, Iterable[str]]] = None,
               cmd_labels: Optional[Union[str, Iterable[str]]] = None,
               trigger_label: Optional[str] = None,
               streamer: bool = False,
               initial_cmd: Optional[Union[Any, Iterable[Any]]] = None,
               exit_cmd: Optional[Union[Any, Iterable[Any]]] = None,
               make_zero_delay: Optional[float] = None,
               ft232h_ser_num: Optional[str] = None,
               spam: bool = False,
               freq: Optional[float] = 200,
               display_freq: bool = False,
               debug: Optional[bool] = False,
               **kwargs) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      name: The name of the :class:`~crappy.inout.InOut` class to instantiate.
      labels: An iterable (e.g. a :obj:`list` or a :obj:`tuple`) containing the
        output labels for InOuts that acquire data. They correspond to the
        values returned by the InOut's :meth:`~crappy.inout.InOut.get_data`
        method, so there should be as many labels as returned values, and given
        in the appropriate order. The first label must always be the time
        label, preferably called ``'t(s)'``. This argument can be omitted if
        :meth:`~crappy.inout.InOut.get_data` returns a :obj:`dict`. Ignored if
        the Block has no output Link.
      cmd_labels: An iterable (e.g. a :obj:`list` or a :obj:`tuple`) containing
        the labels considered as inputs of this Block, for InOuts that set
        commands. The values received from these labels will be passed to the
        InOut's :meth:`~crappy.inout.InOut.set_cmd` method, in the same order
        as the labels are given. Usually, time is not part of the
        ``cmd_labels``. Ignored if the Block has no input Link.
      trigger_label: If given, the Block will only read data whenever a value
        is received on this label (can be any value). Ignored if the Block has
        no output Link. A trigger label can also be a cmd label.

        .. versionchanged:: 1.5.10 renamed from *trigger* to *trigger_label*
      streamer: If :obj:`False`, the :meth:`~crappy.inout.InOut.get_data`
        method of the InOut is called for acquiring data, else it is the
        :meth:`~crappy.inout.InOut.get_stream` method. Refer to the
        documentation of these methods for more information.
      initial_cmd: An initial command for the InOut, set during
        :meth:`prepare`. If given, there must be as many values as in
        ``cmd_labels``. Must be given as an iterable (e.g. a :obj:`list` or a
        :obj:`tuple`).
      exit_cmd: A final command for the InOut, set during :meth:`finish`. If
        given, there must be as many values as in ``cmd_labels``. Must be given
        as an iterable (e.g. a :obj:`list` or a :obj:`tuple`).

        .. versionchanged:: 1.5.10 renamed from *exit_values* to *exit_cmd*
      make_zero_delay: If set, will acquire data before the beginning of the
        test and use it to offset all the labels to zero. The data will be
        acquired during the given number of seconds. Ignored if the Block has
        no output Links. Does not work for InOuts that acquire values other
        than numbers (:obj:`str` for example).
        
        .. versionadded:: 1.5.10
      spam: If :obj:`False`, the Block will call
        :meth:`~crappy.inout.InOut.set_cmd` on the InOut object only if the
        current command is different from the previous. Otherwise, it will call
        the method each time a command is received.
      freq: The target looping frequency for the Block. If :obj:`None`, loops 
        as fast as possible.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block while running.
        
        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
        
        .. versionadded:: 2.0.0
      **kwargs: The arguments to be passed to the :class:`~crappy.inout.InOut`.
    """

    self._device: Optional[InOut] = None
    self._ft232h_args = None
    self._read: bool = False
    self._write: bool = False

    super().__init__()
    self.niceness = -10
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    # The label argument can be omitted for streaming
    if labels is None and streamer:
      self.labels = ['t(s)', 'stream']
    # Forcing the labels into a list
    elif labels is not None and isinstance(labels, str):
      self.labels = [labels]
    elif labels is not None:
      self.labels = list(labels)
    else:
      self.labels = None

    # Forcing the cmd_labels into a list or None
    if cmd_labels is not None and isinstance(cmd_labels, str):
      self._cmd_labels = [cmd_labels]
    elif cmd_labels is not None:
      self._cmd_labels = list(cmd_labels)
    else:
      self._cmd_labels = None

    # Forcing the initial_cmd into a list
    if initial_cmd is not None and isinstance(initial_cmd, str):
      self._initial_cmd = [initial_cmd]
    elif initial_cmd is not None:
      self._initial_cmd = list(initial_cmd)
    else:
      self._initial_cmd = None

    # Forcing the exit_cmd into a list
    if exit_cmd is not None and isinstance(exit_cmd, str):
      self._exit_cmd = [exit_cmd]
    elif exit_cmd is not None:
      self._exit_cmd = list(exit_cmd)
    else:
      self._exit_cmd = None

    # Checking that the initial_cmd and exit_cmd length are consistent
    if self._cmd_labels is not None:
      if self._initial_cmd is not None \
          and len(self._initial_cmd) != len(self._cmd_labels):
        raise ValueError("There should be as many values in initial_cmd as "
                         "there are in cmd_labels !")
      if self._exit_cmd is not None \
          and len(self._exit_cmd) != len(self._cmd_labels):
        raise ValueError("There should be as many values in exit_cmd as "
                         "there are in cmd_labels !")

    self._trig_label = trigger_label

    # Checking for deprecated names
    if name in deprecated_inouts:
      raise NotImplementedError(
          f"The {name} InOut was deprecated in version 2.0.0, and renamed "
          f"to {deprecated_inouts[name]} ! Please update your code "
          f"accordingly and check the documentation for more information")

    # Checking that all the given actuators are valid
    if name not in inout_dict:
      possible = ', '.join(sorted(inout_dict.keys()))
      raise ValueError(f"Unknown InOut type : {name} ! "
                       f"The possible types are : {possible}")

    self._io_name = name
    self._inout_kwargs = kwargs

    self._streamer = streamer
    self._spam = spam
    self._make_zero_delay = make_zero_delay

    self._stream_started = False
    self._last_cmd = None
    self._prev_values = dict()

    # Checking whether the InOut communicates through an FT232H
    if inout_dict[self._io_name].ft232h:
      self._ft232h_args = USBServer.register(ft232h_ser_num)

  def prepare(self) -> None:
    """Checks the consistency of the Link layout, opens the InOut and sets the
    initial command if required.

    This method mainly calls the :meth:`~crappy.inout.InOut.open` method of the
    driven InOut.
    """

    # Instantiating the device in a regular way
    if self._ft232h_args is None:
      self._device = inout_dict[self._io_name](**self._inout_kwargs)
    # Instantiating the device and the connection to the FT232H
    else:
      self.log(logging.INFO, "The InOut to open communicates over an FT232H")
      self._device = inout_dict[self._io_name](**self._inout_kwargs,
                                               _ft232h_args=self._ft232h_args)

    # Checking that the block has inputs or outputs
    if not self.inputs and not self.outputs:
      raise IOError('Error ! The IOBlock is neither an input nor an output !')

    # cmd_labels must be defined when the block has inputs
    if self.inputs and self._cmd_labels is None and self._trig_label is None:
      raise ValueError('Error ! The IOBlock has incoming links but no '
                       'cmd_labels have been given !')

    self._read = bool(self.outputs)
    self._write = bool(self._cmd_labels)

    # Now opening the device
    self.log(logging.INFO, f"Opening the {type(self._device).__name__} InOut")
    self._device.open()
    self.log(logging.INFO, f"{type(self._device).__name__} InOut opened")

    # Acquiring data for offsetting the output
    if self._read and self._make_zero_delay is not None:
      self.log(logging.INFO, f"Performing offsetting on the "
                             f"{type(self._device).__name__} InOut")
      self._device.make_zero(self._make_zero_delay)

    # Writing the first command before the beginning of the test if required
    if self._write and self._initial_cmd is not None:
      self.log(logging.INFO, f"Sending the initial command to the "
                             f"{type(self._device).__name__} InOut")
      self._device.set_cmd(*self._initial_cmd)
      self._last_cmd = self._initial_cmd
      self._prev_values |= zip(self._cmd_labels, self._initial_cmd)

  def loop(self) -> None:
    """Reads data from the InOut and/or sets the received commands.

    Data is read from the InOut **only** if this Block has outgoing Links. If
    the ``trigger_label`` is given, data is read only if a trigger is received
    over the given trigger label.

    A command is set on the InOut **only** if this Block has incoming Links,
    and if data is received over these Links. Depending on the value of the
    ``spam`` argument, a command might not be set if it is similar to the
    previous one.

    The data is read from the InOut either by calling its
    :meth:`~crappy.inout.InOut.return_data` or its
    :meth:`~crappy.inout.InOut.return_stream` method, depending if the
    ``streamer`` argument is :obj:`True` of :obj:`False`. The commands are
    always set by calling the :meth:`~crappy.inout.InOut.set_cmd` method.
    """

    # Receiving all the latest data waiting in the links
    data = self.recv_last_data(fill_missing=False)

    # Reading data from the device if there's no trig_label or if data has been
    # received on this trig_label
    if self._read:
      if self._trig_label is None:
        self._read_data()
      elif self._trig_label in data:
        self.log(logging.DEBUG, "Software trigger signal received")
        self._read_data()

    # If no data was received, there's nothing to write
    if not data:
      return

    if self._write:
      # The missing values are completed here, because the trig label must not
      # be artificially created
      self._prev_values |= data
      data |= self._prev_values

      # Keeping only the labels in cmd_labels
      data = {key: val for key, val in data.items() if key in self._cmd_labels}

      # If not all cmd_labels have a value, returning without calling set_cmd
      if len(data) != len(self._cmd_labels):
        self.log(logging.WARNING, f"Not enough values received in the "
                                  f"{type(self._device).__name__} InOut to"
                                  f" set the cmd, cmd not set !")
        return

      # Grouping the command values in a list before passing them to set_cmd
      cmd = [data[label] for label in self._cmd_labels]

      # Setting the command if it's different from the previous or spam is True
      if cmd != self._last_cmd or self._spam:
        self.log(logging.DEBUG, f"Writing the command {cmd} to the "
                                f"{type(self._device).__name__} InOut")
        self._device.set_cmd(*cmd)
        self._last_cmd = cmd

  def finish(self) -> None:
    """Stops the stream, sets the exit command if necessary, and closes the
    InOut.

    This method mainly calls the :meth:`~crappy.inout.InOut.close` method of
    the driven InOut.
    """

    # Stopping the stream
    if self._streamer and self._device is not None:
      self.log(logging.INFO, f"Stopping stream on the "
                             f"{type(self._device).__name__} InOut")
      self._device.stop_stream()

    # Setting the exit command
    if self._write and self._exit_cmd is not None and self._device is not None:
      self.log(logging.INFO, f"Sending the exit command to the "
                             f"{type(self._device).__name__} InOut")
      self._device.set_cmd(*self._exit_cmd)

    # Closing the device
    if self._device is not None:
      self.log(logging.INFO, f"Closing the {type(self._device).__name__} "
                             f"InOut")
      self._device.close()
      self.log(logging.INFO, f"{type(self._device).__name__} InOut closed")

  def _read_data(self) -> None:
    """Reads the data or the stream, offsets the timestamp and sends the data
    to downstream Blocks."""

    if self._streamer:
      # Starting the stream if needed
      if not self._stream_started:
        self.log(logging.INFO, f"Starting stream on the "
                               f"{type(self._device).__name__} InOut")
        self._device.start_stream()
        self._stream_started = True
      # Actually getting the stream
      data = self._device.return_stream()
    else:
      # Regular reading of data
      data = self._device.return_data()

    self.log(logging.DEBUG, f"Read values {data} from the "
                            f"{type(self._device).__name__} InOut")

    if data is None:
      return

    # Making time relative to the beginning of the test
    if isinstance(data, dict) and 't(s)' in data:
      data['t(s)'] -= self.t0
    else:
      data[0] -= self.t0

    self.send(data)
