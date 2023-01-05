# coding: utf-8

from time import time, sleep
from typing import Optional, Dict, Any, Union, List
import numpy as np
import logging
from multiprocessing import current_process

from .._global import DefinitionError


class MetaIO(type):
  """Metaclass ensuring that two InOuts don't have the same name, and that all
  InOuts define the required methods. Also keeps track of all the InOut
  classes, including the custom user-defined ones."""

  classes = {}

  def __new__(mcs, name: str, bases: tuple, dct: dict) -> type:
    return super().__new__(mcs, name, bases, dct)

  def __init__(cls, name: str, bases: tuple, dct: dict) -> None:
    super().__init__(name, bases, dct)

    # Checking that an InOut with the same name doesn't already exist
    if name in cls.classes:
      raise DefinitionError(f"The {name} class is already defined !")

    # Saving the class
    if name != 'InOut':
      cls.classes[name] = cls


class InOut(metaclass=MetaIO):
  """Base class for all InOut objects. Implements methods shared by all the
  these objects, and ensures their dataclass is MetaIO."""

  def __init__(self) -> None:
    """Sets the attributes."""

    self._compensations = list()
    self._logger: Optional[logging.Logger] = None

  def log(self, level: int, msg: str) -> None:
    """"""

    if self._logger is None:
      self._logger = logging.getLogger(
        f"crappy.{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)

  def open(self) -> None:
    """This method should perform any action that's required for initializing
    the hardware and the communication with it.

    Communication with hardware should be avoided in the :meth:`__init__`
    method, and this method is where it should start happening. This method is
    called after Crappy's processes start, i.e. when the associated IOBlock
    already runs separately from all the other blocks.
    """

    ...

  def get_data(self) -> Optional[Union[list, Dict[str, Any]]]:
    """This method should acquire data from a device and return it in a
    :obj:`list` along with a timestamp.

    The timestamp must always be the first returned value, and there can be any
    number of other acquired channels. The same number of values should always
    be returned, and they should be in the same order.

    Alternatively, the values can be returned in a :obj:`dict`. In that case,
    the ``labels`` argument of the IOBlock is ignored and the returned labels
    correspond to the keys of the dict.

    It is alright for this method to return :obj:`None` if there's no data to
    acquire.
    """

    self.log(logging.WARNING,
             "The get_data method was called but is not defined !\n To get "
             "rid of this warning, define a get_data method, or remove all "
             "the downstream links and make sure the make_zero_delay argument "
             "of the IOBlock is set to None")
    sleep(1)
    return

  def set_cmd(self, *cmd) -> None:
    """This method should handle commands received from the upstream blocks.

    Usually the command is meant to be set on a device, but any other behavior
    is possible. The commands will be passed to this method as `args` (not
    `kwargs`), in the same order as the ``cmd_labels`` are given in the
    IOBlock.

    If the expected number of commands is always the same, you can simply put
    as many `args` to your ``set_cmd`` method as there are commands. For
    example for three commands:
    ::

      def set_cmd(self, cmd0, cmd1, cmd2):
        ...

    Alternatively, or if the number of commands may vary from one test to
    another, you can get all the commands at once in a :obj:`tuple` by putting
    a single unpacking argument. Example:
    ::

      def set_cmd(self, *cmds):
        number_of_commands = len(cmds)
        cmd0 = cmds[0]
        ...

    """

    self.log(logging.WARNING,
             "The set_cmd method was called but is not defined ! The data "
             "received from the incoming links is discarded")
    sleep(1)
    return

  def start_stream(self) -> None:
    """This method should start the acquisition of the stream."""

    self.log(logging.WARNING, "The start_stream method was called but is not "
                              "defined !")

  def get_stream(self) -> Optional[List[np.ndarray]]:
    """This method should acquire a stream as a numpy array, and return it in a
    :obj:`list` along with an array carrying the timestamps.

    The time array must be the first element of the list, the stream array the
    second element. The time array should have only one column, the stream
    array can have any number of columns representing the different channels
    acquired.

    It is also possible to return the two arrays in a :obj:`dict`, in which
    case the ``labels`` argument is ignored and the keys of the dict set the
    returned labels.

    It is alright for this method to return :obj:`None` if there's no data to
    acquire.
    """

    self.log(logging.WARNING, "The get_stream method was called but is not "
                              "defined ! No data sent to downstream links")
    sleep(1)
    return

  def stop_stream(self) -> None:
    """This method should stop the acquisition of the stream."""

    self.log(logging.WARNING, "The stop_stream method was called but is not "
                              "defined !")

  def close(self) -> None:
    """This method should perform any action required for properly ending the
    test and closing the communication with hardware.

    It will be called when the associated IOBlock receives the order to stop,
    either because the user hit CTRL+C, or because a Generator block reached
    the end of its path, or because an exception was raised in any of the
    blocks.
    """

    ...

  def make_zero(self, delay: float) -> None:
    """Acquires data for a given delay, averages it for each channel, and
    stores the average.

    Does not work for pure streams, as it requires a :meth:`get_data` for
    acquiring the data.

    The average values will then be used to remove the offset of the acquired
    data during the test.
    """

    buf = []
    t0 = time()

    # Acquiring data for a given delay
    while time() < t0 + delay:
      data = self.get_data()
      if data is not None and len(data) > 1:
        buf.append(data[1:])

    # If no data could be acquired, abort
    if not buf:
      self.log(logging.WARNING, "No data acquired when zeroing the channels, "
                                "aborting the zeroing")
      return

    # Averaging the values and storing them
    for values in zip(*buf):
      try:
        self._compensations.append(-sum(values) / len(values))
      except TypeError:
        # If something goes wrong, just forget about the offsetting
        self._compensations = list()
        self.log(logging.WARNING,
                 "Cannot calculate the offset !\nPossible reasons are that the"
                 " InOut doesn't return only numbers, or that it returns a "
                 "dict instead of the expected list")
        return

  def return_data(self) -> Optional[Union[list, Dict[str, Any]]]:
    """Returns the data from :meth:`get_data`, corrected by an offset if the
    ``make_zero_delay`` argument of the IOBlock is set."""

    data = self.get_data()

    # If there's no offsetting, just return the data
    if data is None or not self._compensations:
      return data

    # Otherwise, offset the acquired data except for time
    elif len(data[1:]) == len(self._compensations):
      try:
        return [data[0]] + [val + comp for val, comp in
                            zip(data[1:], self._compensations)]
      # Shouldn't happen but doesn't harm to be careful
      except TypeError:
        return data

    # Should also not happen
    else:
      raise ValueError("The number of offsets doesn't match the number of "
                       "acquired values.")

  def return_stream(self) -> Optional[List[np.ndarray]]:
    """Returns the data from :meth:`get_stream`, corrected by an offset if the
    ``make_zero_delay`` argument of the IOBlock is set."""

    data = self.get_stream()

    # If there's no offsetting, just return the data
    if data is None or not self._compensations:
      return data

    # Otherwise, offset the acquired data except for time
    elif data[1].shape[1] == len(self._compensations):
      try:
        return [data[0], data[1] + self._compensations]
      # Shouldn't happen but doesn't harm to be careful
      except TypeError:
        return data

    # There's a problem with the shape of the output data
    else:
      raise ValueError("The number of offsets doesn't match the shape of the "
                       "acquired array.")
