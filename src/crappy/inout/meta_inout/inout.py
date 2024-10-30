# coding: utf-8

from time import time, sleep
from typing import Optional, Any, Union
from collections.abc import Iterable
import numpy as np
import logging
from multiprocessing import current_process
from collections import defaultdict

from .meta_inout import MetaIO


class InOut(metaclass=MetaIO):
  """Base class for all InOut objects. Implements methods shared by all the
  InOuts, and ensures their dataclass is MetaIO.

  The InOut objects are helper classes used by the
  :class:`~crappy.blocks.IOBlock` to interface with hardware.
  
  .. versionadded:: 1.4.0
  """

  ft232h: bool = False

  def __init__(self, *_, **__) -> None:
    """Sets the attributes.
    
    .. versionchanged:: 2.0.0 now accepts args and kwargs
    """

    self._compensations: list[float] = list()
    self._compensations_dict: dict[str, float] = dict()
    self._logger: Optional[logging.Logger] = None

  def log(self, level: int, msg: str) -> None:
    """Records log messages for the InOut.

    Also instantiates the logger when logging the first message.

    Args:
      level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.
    
    .. versionadded:: 2.0.0
    """

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)

  def open(self) -> None:
    """This method should perform any action that's required for initializing
    the hardware and the communication with it.

    Communication with hardware should be avoided in the :meth:`__init__`
    method, and this method is where it should start. This method is called
    after Crappy's processes start, i.e. when the associated
    :class:`~crappy.blocks.IOBlock` already runs separately from all the other
    Blocks.

    It is fine for this method not to perform anything.

    .. versionadded:: 1.5.10
    """

    ...

  def get_data(self) -> Optional[Union[Iterable, dict[str, Any]]]:
    """This method should acquire data from a device and return it along with a
    timestamp.

    The data can be either returned as raw values in an iterable object (like
    a :obj:`list` or a :obj:`tuple`), or along with labels in a :obj:`dict`.
    Values can be of any type (:obj:`float`, :obj:`int`, :obj:`str`,
    :obj:`bytes`, etc.), but most Blocks in Crappy will expect numeric values.

    If the data is returned in a dictionary, its keys will be the labels for
    sending data to downstream Blocks. The ``labels`` argument of the
    :class:`~crappy.blocks.IOBlock` is ignored. It is **mandatory** to use
    `'t(s)'` as the label for the time value.

    Example:
      ::

        return {'t(s)': time.time(), 'value_1': -3.5, 'value_2': 4.1}

    If the data is returned as another type of iterable, the labels must be
    provided in the ``labels`` argument of the :class:`~crappy.blocks.IOBlock`.
    The number of labels must match the number of returned values. It is
    **mandatory** to return the time value first in the iterable.

    Example:
      ::

        return time.time(), -3.5, 4.1

    In both cases, there's no limit to the number of returned values. The same
    number of values must always be returned, and each value must keep a
    consistent type throughout the test.

    It is alright for this method to return :obj:`None` if there's no data to
    acquire.
    
    .. versionadded:: 1.5.10
    """

    self.log(logging.WARNING,
             "The get_data method was called but is not defined !\n To get "
             "rid of this warning, define a get_data method, or remove all "
             "the downstream links and make sure the make_zero_delay argument "
             "of the IOBlock is set to None")
    sleep(1)
    return

  def set_cmd(self, *cmd) -> None:
    """This method should handle commands received from the upstream Blocks.

    Among all the data received on the incoming Link(s), only the labels listed
    in the ``cmd_labels`` argument of the :class:`~crappy.blocks.IOBlock` are
    considered. The command values are passed as arguments to this method in
    the same order as the ``cmd_labels`` are given. They are passed as
    positional arguments, not keyword arguments.

    Example:
      If ``{'t(s)': 20.5, 'value_1': 1.5, 'value_2': 3.6, 'value_3': 'OK'}`` is
      received from upstream Blocks, and ``cmd_labels=('value_3', 'value_1')``,
      then the call will be ``set_cmd('OK', 1.5)``.

    .. versionadded:: 1.5.10
    """

    self.log(logging.WARNING,
             "The set_cmd method was called but is not defined ! The data "
             "received from the incoming links is discarded")
    sleep(1)
    return

  def start_stream(self) -> None:
    """This method should start the acquisition of the stream.

    .. versionadded:: 1.5.10
    """

    self.log(logging.WARNING, "The start_stream method was called but is not "
                              "defined !")

  def get_stream(self) -> Optional[Union[Iterable[np.ndarray],
                                         dict[str, np.ndarray]]]:
    """This method should acquire a stream as a :obj:`numpy.array`, and return
    it along with another array carrying the timestamps.

    Two arrays should be created: one of dimension `(m,)` carrying the
    timestamps, and one of dimension `(m, n)` carrying the acquired data. They
    represent `n` channels acquired at `m` successive time points. There's no
    maximum number of acquired channels, but this number must be constant
    throughout a test.

    The two arrays can either be returned as is in an iterable object (like a
    :obj:`list` or a :obj:`tuple`), or along with labels in a :obj:`dict`.

    If the arrays are returned in a dictionary, its keys will be the labels for
    sending data to downstream Blocks. The ``labels`` argument of the
    :class:`~crappy.blocks.IOBlock` is ignored. It is **mandatory** to use
    `'t(s)'` as the label for the time array.

    If the arrays are returned in another type of iterable, two labels must be
    provided in the ``labels`` argument of the :class:`~crappy.blocks.IOBlock`.
    The first label **must be** `'t(s)'`, the second can be anything. It is
    **mandatory** to return the time value first in the iterable.

    It is alright for this method to return :obj:`None` if there's no data to
    acquire.

    Note:
      It is technically  possible to return more than two arrays, or even
      objects that are not arrays, but it is not the intended use for this
      method and might interfere with some functionalities of Crappy.

    .. versionadded:: 1.5.10
    """

    self.log(logging.WARNING, "The get_stream method was called but is not "
                              "defined ! No data sent to downstream links")
    sleep(1)
    return

  def stop_stream(self) -> None:
    """This method should stop the acquisition of the stream.

    .. versionadded:: 1.5.10
    """

    self.log(logging.WARNING, "The stop_stream method was called but is not "
                              "defined !")

  def close(self) -> None:
    """This method should perform any action required for properly ending the
    test and closing the communication with hardware.

    It will be called when the associated :class:`~crappy.blocks.IOBlock`
    receives the order to stop (usually because the user hit `CTRL+C`, or
    because a :class:`~crappy.blocks.Generator` Block reached the end of its
    path, or because an exception was raised in any of the Blocks).

    It is fine for this method not to perform anything.

    .. versionadded:: 1.5.10
    """

    ...

  def make_zero(self, delay: float) -> None:
    """This method acquires data for a given delay and stores for each channel
    the average value.

    These values are used for zeroing the channels, so that their values start
    at `0` in the beginning of the test.

    Important:
      This method uses :meth:`get_data` for acquiring the values, so it doesn't
      work for pure streams. It also doesn't work if the acquired values do not
      support arithmetic operations (like :obj:`str`).

    .. versionadded:: 1.5.10
    """

    buf = []
    buf_dict = defaultdict(list)
    t0 = time()

    # Acquiring data for a given delay
    while time() < t0 + delay:
      data = self.get_data()
      if data is not None:

        # Case when the InOut returns a dict
        if isinstance(data, dict):
          for label, value in data.items():
            buf_dict[label].append(value)

        # Case when the InOut returns another type of iterable
        else:
          data = list(data)
          if len(data) > 1:
            buf.append(data[1:])

    # Removing the time information if given
    if 't(s)' in buf_dict:
      del buf_dict['t(s)']

    # If no data could be acquired, abort
    if not buf and not buf_dict:
      self.log(logging.WARNING, "No data acquired when zeroing the channels, "
                                "aborting the zeroing")
      return

    # Averaging the values and storing them in a list
    if buf:
      for values in zip(*buf):
        try:
          self._compensations.append(-sum(values) / len(values))
        except TypeError:
          # If something goes wrong, just forget about the offsetting
          self._compensations = list()
          self.log(logging.WARNING,
                   "Cannot calculate the offset !\nA possible reason is that "
                   "the InOut doesn't return only numbers")
          return

    # Averaging the values and storing them in a dict
    else:
      for label, values in buf_dict.items():
        try:
          self._compensations_dict[label] = -sum(values) / len(values)
        except TypeError:
          # If something goes wrong, just forget about the offsetting
          self._compensations_dict = dict()
          self.log(logging.WARNING,
                   "Cannot calculate the offset !\nA possible reason is that "
                   "the InOut doesn't return only numbers in the dict")
          return

  def return_data(self) -> Optional[Union[list[Any], dict[str, Any]]]:
    """Returns the data from :meth:`get_data`, corrected by an offset if the
    ``make_zero_delay`` argument of the :class:`~crappy.blocks.IOBlock` is
    set.

    .. versionadded:: 1.5.10
    """

    data = self.get_data()

    # Nothing to do if there's no data
    if data is None:
      return

    # Case when the data is returned as a dict
    if isinstance(data, dict):
      # Checking that the time label is given
      if 't(s)' not in data:
        raise ValueError("The time label 't(s)' must be given in the dict "
                         "returned by get_data() !")

      # If there's no offsetting, just return the data
      if not self._compensations_dict:
        return data

      # Offsetting if all the labels in data have an offset, except time label
      elif all(label in self._compensations_dict
               for label in data if label != 't(s)'):
        t = {'t(s)': data['t(s)']}
        comp = {label: data[label] + self._compensations_dict[label]
                for label in data if label != 't(s)'}
        return dict(**t, **comp)

      else:
        raise ValueError("Not all the labels in the acquired data have an "
                         "offset value !")

    # Case when data is returned as an iterable but not a dict
    else:
      # Converting to list for convenience
      data = list(data)

      # If there's no offsetting, just return the data
      if not self._compensations:
        return data

      # Offsetting if the number of acquired values match the number of offsets
      elif len(data[1:]) == len(self._compensations):
        return [data[0]] + [val + comp for val, comp
                            in zip(data[1:], self._compensations)]

      else:
        raise ValueError("The number of offsets doesn't match the number of "
                         "acquired values.")

  def return_stream(self) -> Optional[Union[list[np.ndarray],
                                            dict[str, np.ndarray]]]:
    """Returns the data from :meth:`get_stream`, corrected by an offset if the
    ``make_zero_delay`` argument of the :class:`~crappy.blocks.IOBlock` is
    set.

    .. versionadded:: 1.5.10
    """

    data = self.get_stream()

    # Nothing to do if there's no data
    if data is None:
      return

    # Case when the stream is returned as a dict
    if isinstance(data, dict):
      # Checking that the time label is given
      if 't(s)' not in data:
        raise ValueError("The time label 't(s)' must be given in the dict "
                         "returned by get_stream() !")

      # If there's no offsetting, just return the stream
      if not self._compensations and not self._compensations_dict:
        return data

      t = {'t(s)': data['t(s)']}
      comp = dict()
      # Offsetting if the shape of the acquired values match the number of
      # offsets
      for label, array in ((key, val) for key, val in data.items()
                           if key != 't(s)'):
        # Case when get_data returns an iterable
        if self._compensations and len(self._compensations) == array.shape[1]:
          comp[label] = array + self._compensations
        # Case when get_data returns a dict
        elif (self._compensations_dict and
              len(self._compensations_dict) == array.shape[1]):
          comp[label] = array + list(self._compensations_dict.values())
        # The shapes do not match
        else:
          raise ValueError("The number of offsets doesn't match the shape of "
                           "the acquired array !")
        return dict(**t, **comp)

    # Case when stream is returned as an iterable but not a dict
    else:
      # Converting to list for convenience
      data = list(data)

      # If there's no offsetting, just return the stream
      if not self._compensations and not self._compensations_dict:
        return data

      # Offsetting if the shape of the acquired values match the number of
      # offsets
      comp = list()
      for array in data[1:]:
        # Case when get_data returns an iterable
        if self._compensations and len(self._compensations) == array.shape[1]:
          comp.append(array + self._compensations)
        # Case when get_data returns a dict
        elif (self._compensations_dict and
              len(self._compensations_dict) == array.shape[1]):
          comp.append(array + self._compensations_dict.values())
        # The shapes do not match
        else:
          raise ValueError("The number of offsets doesn't match the shape "
                           "of the acquired array !")
      return [data[0]] + comp
