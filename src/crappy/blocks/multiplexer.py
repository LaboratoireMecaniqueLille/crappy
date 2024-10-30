# coding: utf-8

import logging
import numpy as np
from typing import Optional, Union
from collections.abc import Iterable
from collections import defaultdict

from .meta_block import Block


class Multiplexer(Block):
  """This Block takes data from upstream Blocks as input and interpolates it to
  output all the labels in a common time basis.

  This Block is very similar to the :class:`~crappy.blocks.Synchronizer` Block,
  but the `Synchronizer` takes the timestamps of a reference label as a time
  base whereas this one performs the interpolation on a time base independent
  of the received labels.

  It can take any number of inputs, provided that they all share a common time
  label. It is also possible to choose which labels are considered for
  multiplexing and which are dropped. The interpolation is performed using the
  :obj:`numpy.interp` method.

  This Block is useful for synchronizing data acquired from different sensors,
  e.g. to plot a real-time stress-strain curve with position data coming from a
  :class:`~crappy.blocks.Machine` Block and force data coming from a
  :class:`~crappy.blocks.IOBlock` Block. Multiplexing is however quite
  resource-consuming, so it is preferable to perform interpolation when
  post-processing if real-time is not needed.

  Note:
    This Block doesn't truly output data in real-time as it needs to wait for
    data from all the upstream Blocks before performing the interpolation. It
    should only be used with care as an input to a decision-making Block. This
    is especially true when the upstream Blocks have very different sampling
    rates.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Multiplex* to *Multiplexer*
  """

  def __init__(self,
               time_label: str = 't(s)',
               out_labels: Optional[Union[str, Iterable[str]]] = None,
               interp_freq: float = 200,
               freq: Optional[float] = 50,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      time_label: The label carrying the time information.
      
        .. versionchanged:: 1.5.10 renamed from *key* to *time_label*
      out_labels: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        the labels to multiplex, except for the time label that is given
        separately in the ``time_label`` argument. The Block also doesn't
        output anything until data has been received on all these labels. If
        left to :obj:`None`, all the received labels are considered. **It is
        recommended to always set this argument !** It is also possible to
        give this argument as a single :obj:`str` (i.e. not in an iterable),
        although multiplexing a single label is of limited interest.
        
        .. versionadded:: 2.0.0
      interp_freq: The target frequency for performing the interpolation. In 
        the output data, there will be one interpolated data point each 
        :math:`1 / interp_freq` seconds. Independent of the ``freq`` argument, 
        but it is no use setting ``freq`` higher than ``interp_freq`` otherwise 
        there will be void loops.
        
        .. versionadded:: 2.0.0
      freq: The target looping frequency for the Block. If :obj:`None`, loops 
        as fast as possible.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
        
        .. versionadded:: 1.5.9
        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
        
        .. versionadded:: 2.0.0
    """

    super().__init__()
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    # Initializing the attributes
    self._time_label = time_label
    self._interp_freq = interp_freq
    self._data: dict[str, np.ndarray] = defaultdict(self._default_array)
    self._delta: float = 1 / self._interp_freq / 20
    self._last_max_t: float = -float('inf')

    # Forcing the out_labels into a list
    if out_labels is not None and isinstance(out_labels, str):
      self._out_labels = [out_labels]
    elif out_labels is not None:
      self._out_labels = list(out_labels)
    else:
      self._out_labels = None

  def loop(self) -> None:
    """Receives data, interpolates it, and sends it to the downstream
    Blocks."""

    # Iterating over all the links
    for link_data in self.recv_all_data_raw():
      # Only data associated with a time label can be multiplexed
      if self._time_label not in link_data:
        continue
      # Extracting the time information from the data
      timestamps = link_data.pop(self._time_label)

      # Adding data from each label in the buffer
      for label, values in link_data.items():
        # Only the labels specified in out_labels is considered
        if self._out_labels is not None and label not in self._out_labels:
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

    # Making sure there's data for all the requested labels
    if (self._out_labels is not None and
        any(label not in self._data for label in self._out_labels)):
      self.log(logging.DEBUG, "Not all the requested labels received yet")
      return

    # There should also be at least two values for each label
    if any(len(self._data[label][0]) < 2 for label in self._data):
      self.log(logging.DEBUG, "Not at least 2 values for each label in buffer")
      return

    # The two values should also be separated by at least one time period
    if any(np.ptp(self._data[label][0]) < 1 / self._interp_freq
           for label in self._data):
      self.log(logging.DEBUG, "At least one label has values too close "
                              "together compared to interpolation frequency")
      return

    # Getting the minimum time for the interpolation (maximin over all labels)
    min_t = max(data[0, 0] for data in self._data.values())
    # The minimum must be higher than the previous maximum
    min_t = max(min_t, self._last_max_t + self._delta)
    # Correcting to the closest upper multiple of the time interval
    min_t = min_t + (1 / self._interp_freq) - min_t % (1 / self._interp_freq)

    # Getting the maximum time for the interpolation (minimax over all labels)
    max_t = min(data[0, -1] for data in self._data.values())
    # Correcting to the closest lower multiple of the time interval
    max_t = max_t - (1 / self._interp_freq) + max_t % (1 / self._interp_freq)

    if max_t < min_t:
      self.log(logging.DEBUG, "Ranges not matching for interpolation")
      return

    # The array containing the timestamps for interpolating
    interp_times = np.arange(min_t, max_t + self._delta, 1 / self._interp_freq)

    # Making sure there are points to interpolate
    if not np.any(interp_times):
      self.log(logging.DEBUG, "No time points for interpolation")
      return

    to_send = dict()
    self._last_max_t = max_t

    # Building the dict of values to send
    for label, values in self._data.items():
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
