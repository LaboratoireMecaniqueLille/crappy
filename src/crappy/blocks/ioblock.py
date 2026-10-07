# coding: utf-8

from typing import Any
from collections.abc import Sequence
import logging
from numbers import Real
from math import isfinite

from .meta_block import Block
from .._collection import (CollectionEntry, collection_registry,
                           load_collection_class)
from ..inout import inout_dict, InOut, moved_to_collection


class IOBlock(Block):
  """This Block is meant to drive :class:`~crappy.inout.meta_inout.inout.InOut`
  objects. It can acquire data, and/or set commands. One IOBlock can only drive
  a single InOut.

  If it has incoming :class:`~crappy.links.link.Link`, it will set the commands
  received over the labels given in ``cmd_labels`` by calling the 
  :meth:`~crappy.inout.meta_inout.inout.InOut.set_cmd` method of the InOut.
  Additional commands to set at the very beginning or the very end of the test
  can also be specified.

  If it has outgoing :class:`~crappy.links.link.Link`, it will acquire data
  using the :meth:`~crappy.inout.meta_inout.inout.InOut.get_data` method of the
  InOut and send it downstream over the labels given in ``labels``. It is
  possible to trigger the acquisition using a predefined label.

  The ``streamer`` argument allows using the "streamer" mode of InOuts
  supporting it, instead of the regular acquisition mode. Finally, the
  ``make_zero_delay`` argument allows offsetting the acquired values to zero at
  the beginning of the test. Refer to the documentation of each argument for a
  more detailed description.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.1.0 removed support for InOuts communicating through
     an FT232H
  """

  def __init__(self,
               name: str,
               labels: str | Sequence[str] | None = None,
               cmd_labels: str | Sequence[str] | None = None,
               trigger_label: str | None = None,
               streamer: bool = False,
               initial_cmd: Sequence[Any] | None = None,
               exit_cmd: Sequence[Any] | None = None,
               make_zero_delay: float | None = None,
               spam: bool = False,
               freq: float | None = 200,
               display_freq: bool = False,
               debug: bool | None = False,
               **kwargs) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      name: The name of the :class:`~crappy.inout.meta_inout.inout.InOut` class
        to instantiate.
      labels: A sequence (e.g. a :obj:`list` or a :obj:`tuple`) containing the
        output labels for InOuts that acquire data. They correspond to the
        values returned by the InOut's
        :meth:`~crappy.inout.meta_inout.inout.InOut.get_data` method, so there
        should be as many labels as returned values, and given in the
        appropriate order. The first label must always be the time label,
        preferably called ``'t(s)'``. This argument can be omitted if
        :meth:`~crappy.inout.meta_inout.inout.InOut.get_data` returns a
        :obj:`dict`. Ignored if the Block has no output Link.
      cmd_labels: A sequence (e.g. a :obj:`list` or a :obj:`tuple`) containing
        the labels considered as inputs of this Block, for InOuts that set
        commands. The values received from these labels will be passed to the
        InOut's :meth:`~crappy.inout.meta_inout.inout.InOut.set_cmd` method, in
        the same order as the labels are given. Usually, time is not part of
        the ``cmd_labels``. Ignored if the Block has no input Link.
      trigger_label: If given, the Block will only read data whenever a value
        is received on this label (can be any value). Ignored if the Block has
        no output Link. A trigger label can also be a cmd label.

        .. versionchanged:: 1.5.10 renamed from *trigger* to *trigger_label*
      streamer: If :obj:`False`, the
        :meth:`~crappy.inout.meta_inout.inout.InOut.get_data` method of the
        InOut is called for acquiring data, else it is the
        :meth:`~crappy.inout.meta_inout.inout.InOut.get_stream` method. Refer
        to the documentation of these methods for more information.
      initial_cmd: An initial command for the InOut, set during
        :meth:`prepare`. If given, there must be as many values as in
        ``cmd_labels``. Must be given as a sequence (e.g. a :obj:`list` or a
        :obj:`tuple`).
      exit_cmd: A final command for the InOut, set during :meth:`finish`. If
        given, there must be as many values as in ``cmd_labels``. Must be given
        as a sequence (e.g. a :obj:`list` or a :obj:`tuple`).

        .. versionchanged:: 1.5.10 renamed from *exit_values* to *exit_cmd*
      make_zero_delay: If set, will acquire data before the beginning of the
        test and use it to offset all the labels to zero. The data will be
        acquired during the given number of seconds. Ignored if the Block has
        no output Links. Does not work for InOuts that acquire values other
        than numbers (:obj:`str` for example).
        
        .. versionadded:: 1.5.10
      spam: If :obj:`False`, the Block will call
        :meth:`~crappy.inout.meta_inout.inout.InOut.set_cmd` on the InOut
        object only if the current command is different from the previous.
        Otherwise, it will call the method each time a command is received.
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
      **kwargs: The arguments to be passed to the
        :class:`~crappy.inout.meta_inout.inout.InOut`.

    .. versionremoved:: 2.1.0 *ft232h_ser_num* argument
    """

    self._device: InOut | None = None
    self._device_opened: bool = False
    self._exit_cmd_sent: bool = False
    self._read: bool = False
    self._write: bool = False

    super().__init__()
    self.niceness = -10
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    match name:
      case str() if name.strip():
        pass
      case str():
        raise ValueError("The InOut name must be provided as a non-empty "
                         "string")
      case _:
        raise TypeError("The InOut name must be provided as a non-empty "
                        "string")

    # None means that this is an ordinary core or user-defined InOut
    self._collection_entry: CollectionEntry | None = None

    # Check if the requested InOut is part of crappy.collection
    entry = collection_registry.get("InOut", name)

    # Checking that the given InOut name is valid
    if name not in inout_dict:
      # First option, the InOut should be loaded from crappy.collection
      if entry is not None:
        # This call raises early if the module cannot be loaded
        load_collection_class(entry, inout_dict)
        self._collection_entry = entry
      # Second case, the InOut was moved to crappy.collection but this module
      # was not imported
      elif name in moved_to_collection:
        raise NotImplementedError(f"The InOut {name} was moved to "
                                  f"crappy.collection. To use it, simply add "
                                  f"import crappy.collection at the beginning "
                                  f"of your script")
      # The name of the InOut simply cannot be found anywhere
      else:
        possible = ', '.join(sorted(inout_dict.keys()))
        raise ValueError(f"Unknown InOut name : {name}! "
                         f"The currently available ones are: {possible}")
    # Case when the InOut was already loaded in a separate Block
    elif entry is not None and inout_dict[name].__module__ == entry.module:
      self._collection_entry = entry

    self._io_name: str = name

    match streamer:
      case bool():
        self._streamer: bool = streamer
      case _:
        raise TypeError("streamer mut be provided as a boolean")

    match labels:
      case None if streamer:
        self.labels = ['t(s)', 'stream']
      case None:
        self.labels = labels
      case ():
        raise ValueError("labels were provided as an empty sequence, set to "
                         "None instead if you don't want to set any labels")
      case str() if labels.strip():
        self.labels = [labels]
      case str():
        raise ValueError("labels were provided as an empty string, set to "
                         "None instead if you don't want to set any labels")
      case (*labels,) if (all(isinstance(label, str) for label in labels) and
                          all(label.strip() for label in labels)):
        self.labels = list(labels)
      case (*labels, ) if all(isinstance(label, str) for label in labels):
        raise ValueError("All the labels must be provided as non-empty "
                         "strings")
      case (*_,):
        raise TypeError("All the labels must be provided as non-empty strings")
      case _:
        raise TypeError("The IOBlock labels must be provided as a sequence of "
                        "non-empty strings, a non-empty strings, or None")

    match cmd_labels:
      case None:
        self._cmd_labels: list[str] = list()
      case ():
        raise ValueError("cmd_labels were provided as an empty sequence, set "
                         "to None instead if you don't want to set any labels")
      case str() if cmd_labels.strip():
        self._cmd_labels: list[str] = [cmd_labels]
      case str():
        raise ValueError("cmd_labels were provided as an empty string, set to "
                         "None instead if you don't want to set any labels")
      case (*cmd_labels,) if (all(isinstance(label, str) for label
                                  in cmd_labels) and
                              all(label.strip() for label in cmd_labels)):
        self._cmd_labels: list[str] = list(cmd_labels)
      case (*cmd_labels, ) if all(isinstance(label, str) for label
                                  in cmd_labels):
        raise ValueError("All the cmd_labels must be provided as non-empty "
                         "strings")
      case (*_,):
        raise TypeError("All the cmd_labels must be provided as non-empty "
                        "strings")
      case _:
        raise TypeError("The IOBlock cmd_labels must be provided as a "
                        "sequence of non-empty strings, a non-empty strings, "
                        "or None")

    match trigger_label:
      case None:
        self._trig_label: str | None = trigger_label
      case str() if trigger_label.strip():
        self._trig_label: str | None = trigger_label
      case str():
        raise ValueError("trigger_label must be provided as a non-empty "
                         "string or None")
      case _:
        raise TypeError("trigger_label must be provided as a non-empty "
                        "string or None")

    match initial_cmd:
      case None:
        self._initial_cmd: list[Any] | None = initial_cmd
      case str() if initial_cmd.strip():
        self._initial_cmd: list[Any] | None = [initial_cmd]
      case str():
        raise ValueError("If provided as a string, initial_cmd must be "
                         "non-empty")
      case ():
        raise ValueError("If provided as a sequence, initial_cmd must be "
                         "non-empty")
      case (*_,):
        self._initial_cmd: list[Any] | None = list(initial_cmd)
      case _:
        raise TypeError("The initial_cmd must be provided as a non-empty "
                        "sequence or None")

    match exit_cmd:
      case None:
        self._exit_cmd: list[Any] | None = exit_cmd
      case str() if exit_cmd.strip():
        self._exit_cmd: list[Any] | None = [exit_cmd]
      case str():
        raise ValueError("If provided as a string, exit_cmd must be "
                         "non-empty")
      case ():
        raise ValueError("If provided as a sequence, exit_cmd must be "
                         "non-empty")
      case (*_,):
        self._exit_cmd: list[Any] | None = list(exit_cmd)
      case _:
        raise TypeError("The exit_cmd must be provided as a non-empty "
                        "sequence or None")

    match make_zero_delay:
      case None:
        self._make_zero_delay: float | None = None
      case Real() if make_zero_delay >= 0 and isfinite(make_zero_delay):
        self._make_zero_delay: float | None = float(make_zero_delay)
      case Real():
        raise ValueError("make_zero_delay must be provided as a positive, "
                         "finite float or None")
      case _:
        raise TypeError("make_zero_delay must be provided as a positive, "
                        "finite float or None")

    match spam:
      case bool():
        self._spam: bool = spam
      case _:
        raise TypeError("spam mut be provided as a boolean")

    # Checking that the initial_cmd and exit_cmd length are consistent
    if self._cmd_labels:
      if (self._initial_cmd is not None
          and len(self._initial_cmd) != len(self._cmd_labels)):
        raise ValueError("There should be as many values in initial_cmd as "
                         "there are in cmd_labels!")
      if (self._exit_cmd is not None
          and len(self._exit_cmd) != len(self._cmd_labels)):
        raise ValueError("There should be as many values in exit_cmd as "
                         "there are in cmd_labels!")

    self._inout_kwargs: dict[str, Any] = kwargs
    self._stream_started: bool = False
    self._last_cmd: list[Any] | None = None
    self._prev_values: dict[str, Any] = dict()

  def prepare(self) -> None:
    """Checks the consistency of the Link layout, opens the InOut and sets the
    initial command if required.

    This method mainly calls the
    :meth:`~crappy.inout.meta_inout.inout.InOut.open` method of the driven
    InOut.
    """

    # Checking that the Block has inputs or outputs
    if not self.inputs and not self.outputs:
      raise IOError('Error ! The IOBlock is neither an input nor an output!')

    # cmd_labels must be defined when the Block has inputs
    if self.inputs and not self._cmd_labels and self._trig_label is None:
      raise ValueError('Error! The IOBlock has incoming links but no '
                       'cmd_labels have been given!')

    self._read = bool(self.outputs)
    self._write = bool(self._cmd_labels)

    # Under the spawn multiprocessing start method, it is necessary to re-load
    # the modules from crappy.collection
    if self._collection_entry is not None:
      load_collection_class(self._collection_entry, inout_dict)

    # Instantiating the device
    self._device = inout_dict[self._io_name](**self._inout_kwargs)

    # Now opening the device
    self.log(logging.INFO, f"Opening the {type(self._device).__name__} InOut")
    self._device.open()
    self._device_opened = True
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
    :meth:`~crappy.inout.meta_inout.inout.InOut.return_data` or its
    :meth:`~crappy.inout.meta_inout.inout.InOut.return_stream` method,
    depending if the ``streamer`` argument is :obj:`True` of :obj:`False`. The
    commands are always set by calling the
    :meth:`~crappy.inout.meta_inout.inout.InOut.set_cmd` method.
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

    This method mainly calls the
    :meth:`~crappy.inout.meta_inout.inout.InOut.close` method of the driven
    InOut.
    """

    # No cleanup to perform is there's no InOut
    if self._device is None:
      return

    name = type(self._device).__name__
    failures: list[Exception | KeyboardInterrupt] = list()

    # Stopping the stream
    if self._streamer and self._stream_started:
      self.log(logging.INFO, f"Stopping stream on the {name} InOut")
      try:
        self._device.stop_stream()
      except (Exception, KeyboardInterrupt) as error:
        error.add_note(f"{name} IOBlock cleanup step: stop stream")
        failures.append(error)
      else:
        self._stream_started = False

    # Setting the exit command independently of stream shutdown
    if (self._device_opened and
        self._write and
        self._exit_cmd is not None and
        not self._exit_cmd_sent):
      self.log(logging.INFO, f"Sending the exit command to the {name} InOut")
      try:
        self._device.set_cmd(*self._exit_cmd)
      except (Exception, KeyboardInterrupt) as error:
        error.add_note(f"{name} IOBlock cleanup step: set exit command")
        failures.append(error)
      else:
        self._exit_cmd_sent = True

    # Closing the device even if earlier cleanup operations failed
    self.log(logging.INFO, f"Closing the {name} InOut")
    try:
      self._device.close()
    except (Exception, KeyboardInterrupt) as error:
      error.add_note(f"{name} IOBlock cleanup step: close device")
      failures.append(error)
    else:
      self._device = None
      self._device_opened = False
      self._stream_started = False
      self.log(logging.INFO, f"{name} InOut closed")

    # If there's only one Exception, raise it
    if len(failures) == 1:
      raise failures[0]
    # Handle the case when a KeyboardInterrupt is among the Exceptions
    elif any(isinstance(error, KeyboardInterrupt) for error in failures):
      for index, error in enumerate(failures):
        if isinstance(error, KeyboardInterrupt):
          others = failures[:index] + failures[index + 1:]
          if error.__cause__ is not None:
            others.insert(0, error.__cause__)
          raise error from BaseExceptionGroup("Other InOut cleanup failures",
                                              others)
    # Otherwise just raise all Exceptions at once
    elif failures:
      raise ExceptionGroup("InOut cleanup failures", failures)

  def _read_data(self) -> None:
    """Reads the data or the stream, offsets the timestamp and sends the data
    to downstream Blocks."""

    assert self._device is not None

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
