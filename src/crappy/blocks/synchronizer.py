# coding: utf-8

import numpy as np
from typing import Optional, Union
from collections.abc import Iterable
from collections import defaultdict
import logging

from .meta_block import Block


class Synchronizer(Block):
  """This Block takes data from upstream Blocks as input and interpolates it to
  output all the labels on the same timestamps as a reference label.

  This Block is very similar to the :class:`~crappy.blocks.Multiplexer` Block,
  but the `Multiplexer` interpolates data in a time base independent of the
  labels whereas this one takes one label as a reference.

  It can take any number of inputs, provided that they all share a common time
  label. It is also possible to choose which labels are considered for
  interpolation and which are dropped. The interpolation is performed using the
  :obj:`numpy.interp` method.

  This Block is useful for synchronizing data acquired from different sensors,
  in the context when one label should be treated as a reference. This is for
  example the case when synchronizing signals with the output of an image
  processing, to be able to compare all the signals in the time base of the
  image acquisition.

  .. versionadded:: 2.0.5
  """

  def __init__(self,
               reference_label: str,
               time_label: str = 't(s)',
               labels_to_sync: Optional[Union[str, Iterable[str]]] = None,
               freq: Optional[float] = 50,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      reference_label: The label whose timestamps will be taken as a time base
        for performing the interpolation.
      time_label: The label carrying the time information. Should be common to
        all the input Blocks.
      labels_to_sync: An iterable (like a :obj:`list` or a :obj:`tuple`)
        containing the labels to interpolate on the reference label's time
        base, except for the time label that is given separately in the
        ``time_label`` argument. The Block also doesn't output anything until
        data has been received on all these labels. If left to :obj:`None`, all
        the received labels are considered. **It is recommended to always set
        this argument !** It is also possible to give this argument as a single
        :obj:`str` (i.e. not in an iterable).
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
    """

    super().__init__()
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    # Initializing the attributes
    self._ref_label = reference_label
    self._time_label = time_label
    self._data: dict[str, np.ndarray] = defaultdict(self._default_array)

    # Forcing the labels_to_sync into a list
    if labels_to_sync is not None and isinstance(labels_to_sync, str):
      self._to_sync = [labels_to_sync]
    elif labels_to_sync is not None:
      self._to_sync = list(labels_to_sync)
    else:
      self._to_sync = None

  def loop(self) -> None:
    """Receives data, interpolates it, and sends it to the downstream
    Blocks."""

    # Iterating over all the links
    for link_data in self.recv_all_data_raw():
      # Only data associated with a time label can be synchronized
      if self._time_label not in link_data:
        continue
      # Extracting the time information from the data
      timestamps = link_data.pop(self._time_label)

      # Adding data from each label in the buffer
      for label, values in link_data.items():
        # Only the labels specified in out_labels is considered
        if (self._to_sync is not None and label not in self._to_sync
            and label != self._ref_label):
          continue

        # Adding the received values to the buffered ones
        self._data[label] = np.concatenate((self._data[label],
                                            np.array((timestamps, values))),
                                           axis=1)
        # Sorting the buffered data, if a same label comes from multiple Links
        self._data[label] = self._data[label][
                            :, self._data[label][0].argsort()]

    # Aborting if there's no data to process
    if not self._data:
      self.log(logging.DEBUG, "No data in the buffer to process")
      return

    # Aborting if there's no data for the reference label
    if self._ref_label not in self._data:
      self.log(logging.DEBUG, "No value for the reference label found in "
                              "the buffer")
      return

    # Making sure there's data for all the requested labels
    if (self._to_sync is not None and
        any(label not in self._data for label in self._to_sync)):
      self.log(logging.DEBUG, "Not all the requested labels received yet")
      return

    # There should also be at least two values for each label
    if any(len(self._data[label][0]) < 2 for label in self._data):
      self.log(logging.DEBUG, "Not at least 2 values for each label in buffer")
      return

    # Getting the minimum time for the interpolation (maximin over all labels)
    min_t = max(data[0, 0] for data in self._data.values())

    # Getting the maximum time for the interpolation (minimax over all labels)
    max_t = min(data[0, -1] for data in self._data.values())

    # Checking if there's a valid time range for interpolation
    if max_t < min_t:
      self.log(logging.DEBUG, "Ranges not matching for interpolation")
      return

    # The array containing the timestamps for interpolating
    interp_times = self._data[self._ref_label][0,
      (self._data[self._ref_label][0] >= min_t) &
      (self._data[self._ref_label][0] <= max_t)]

    # Checking if there are values for the target label in the valid time range
    if not np.any(interp_times):
      self.log(logging.DEBUG,
               "No value of the target label found between the minimum and "
               "maximum possible interpolation times")
      return

    to_send = dict()

    # Building the dict of values to send
    for label, values in self._data.items():

      # Keeping the values of the reference label as they are
      if label == self._ref_label:
        to_send[label] = values[1, :]
      # For all the other labels, performing interpolation
      else:
        to_send[label] = list(np.interp(interp_times, values[0], values[1]))

      # Keeping the last data point before max_t to pass this information on
      last = values[:, values[0] <= max_t][:, -1]
      # Removing the used values from the buffer, except the last data point
      self._data[label] = np.column_stack((last, values[:, values[0] > max_t]))

    if to_send:
      # Adding the time values to the dict of values to send
      to_send[self._time_label] = list(interp_times)

      # Sending the values
      for i, _ in enumerate(interp_times):
        self.send({label: values[i] for label, values in to_send.items()})

  @staticmethod
  def _default_array() -> np.ndarray:
    """Helper function for the default dict."""

    return np.array(([], []))
