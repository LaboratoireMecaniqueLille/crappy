# coding: utf-8

import numpy as np
from typing import Dict, Optional, Iterable, Union
from collections import defaultdict

from .meta_block import Block


class Multiplexer(Block):
  """This block takes data from upstream blocks as input and interpolates it to
  output all labels in a common time basis.

  It is useful for synchronizing data acquired from different sensors, e.g. to
  plot a real-time stress-strain curve. This block is however quite
  resource-consuming, so it is preferable to perform interpolation in
  post-processing if real-time is not needed.

  Note:
    This block doesn't truly output data in real-time as it needs to wait for
    data from all the upstream blocks before performing the interpolation.
    So it should only be used with care when it is an input of a
    decision-making block. This is especially true when the upstream blocks
    have very different sample rates.
  """

  def __init__(self,
               time_label: str = 't(s)',
               out_labels: Optional[Union[str, Iterable[str]]] = None,
               interp_freq: float = 200,
               freq: Optional[float] = 50,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the args and initializes the parent class.

    Args:
      time_label: The label carrying the time information.
      out_labels: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        the labels to multiplex, except for the time label that is given
        separately in the ``time_label`` argument. The Block also doesn't
        output anything until data has been received on all these labels. If
        left to :obj:`None`, all the received labels are considered. **It is
        recommended to always set this argument !** It is also possible to
        give this argument as a single :obj:`str` (i.e. not in an iterable),
        although multiplexing a single label is of limited interest.
      freq : The sample rate for the interpolation, and the target looping
        frequency for the block. If this value is set too high and your machine
        cannot keep up, the block will most likely lag.
      display_freq: If :obj:`True`, displays information about the looping
        frequency of the block.
    """

    super().__init__()
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    # Initializing the attributes
    self._time_label = time_label
    self._interp_freq = interp_freq
    self._data: Dict[str, np.ndarray] = defaultdict(self._default_array)

    # Forcing the out_labels into a list
    if out_labels is not None and isinstance(out_labels, str):
      self._out_labels = [out_labels]
    elif out_labels is not None:
      self._out_labels = list(out_labels)
    else:
      self._out_labels = None

  def loop(self) -> None:
    """Receives data, interpolates it, and sends it to the downstream
    blocks."""

    # Receiving all the upcoming data
    data = self.recv_all_data_raw()

    # Iterating over all the links
    for link_data in data:
      # Only data associated with a time label can be multiplexed
      if self._time_label not in link_data:
        continue
      # Extracting the time information from the data
      timestamps = link_data[self._time_label]

      # Adding data from each label in the buffer
      for label, values in link_data.items():
        # The time information is handled differently
        if label == self._time_label:
          continue
        # Only the labels specified in out_labels is considered
        elif self._out_labels is not None and label not in self._out_labels:
          continue

        # Adding the received values to the buffered ones
        self._data[label] = np.concatenate((self._data[label],
                                            np.array((timestamps, values))),
                                           axis=1)
        # Sorting the buffered data, if a same label comes from multiple Links
        self._data[label] = self._data[label][
          :, self._data[label][0].argsort()]

    # Making sure there's data for all the requested labels
    if self._out_labels is not None:
      if not all(label in self._data for label in self._out_labels):
        return
      elif not all(np.any(self._data[label]) for label in self._out_labels):
        return

    # Making sure there's data for all the labels
    if not self ._data or all(not np.any(data)
                              for data in self._data.values()):
      return

    # Getting the minimum time for the interpolation
    min_t = min(np.min(data[0]) for data in self._data.values())
    # Correcting to the closest lower multiple of the time interval
    min_t = min_t - min_t % (1 / self._interp_freq)

    # Getting the maximum time for the interpolation
    max_t = min(np.max(data[0]) for data in self._data.values())

    # The array containing the timestamps for interpolating
    interp_times = np.arange(min_t, max_t, 1 / self._interp_freq)

    # Correcting to the closest lower multiple of the time interval
    max_t = max_t - max_t % (1 / self._interp_freq)

    # Making sure there are points to interpolate
    if not np.any(interp_times):
      return

    to_send = dict()

    # Building the dict of values to send
    for label, values in self._data.items():
      to_send[label] = list(np.interp(interp_times, values[0], values[1]))
      # Removing the used values from the buffer
      self._data[label] = values[:, values[0] > max_t]

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
