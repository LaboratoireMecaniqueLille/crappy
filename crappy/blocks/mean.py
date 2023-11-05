# coding: utf-8

import numpy as np
from typing import List, Optional
from warnings import warn

from .block import Block


class Mean_block(Block):
  """This block computes the average values over a given delay of each label
  received, and returns them.

  It can take any number of inputs, provided that they share a common time
  label. If the same label (except time) is received from several blocks, it
  may lead to unexpected results.

  Warning:
    If the delay for averaging is too short compared with the looping frequency
    of the upstream blocks, this block may not always return the same number of
    labels ! This can cause errors in downstream blocks expecting a fixed
    number of labels.
  """

  def __init__(self,
               delay: float,
               time_label: str = 't(s)',
               out_labels: Optional[List[str]] = None,
               verbose: bool = False,
               freq: float = 50) -> None:
    """Sets the args and initializes the parent class.

    Args:
      delay: The averaged data will be sent each ``delay`` seconds.
      time_label: The label containing the time information.
      out_labels: If given, only the listed labels and the time will be
        returned. Otherwise, all of them are returned.
      verbose: If :obj:`True`, prints the looping frequency of the block.
      freq: The block will try to loop at this frequency.
    """
    
    if verbose:
      warn("The verbose argument will be replaced by display_freq and debug "
           "in version 2.0.0", FutureWarning)
    warn("The Mean_block Block will be renamed to MeanBlock in version 2.0.0",
         FutureWarning)

    super().__init__()
    self.verbose = verbose
    self.freq = freq

    self._delay = delay
    self._time_label = time_label
    self._out_labels = out_labels

  def prepare(self) -> None:
    """Initializes the buffer and the time counters."""

    self._buffer = {link: dict() for link in self.inputs}
    self._last_sent_t = -self._delay
    self._last_recv_t = 0

  def loop(self) -> None:
    """Receives all available data from the upstream blocks, and averages it
    and sends it if the time delay is reached."""

    # Receiving data from each incoming link
    for link in self.inputs:
      data = link.recv_chunk(blocking=False)

      if data is not None:
        # Updating the last received time attribute
        if self._time_label in data:
          self._last_recv_t = max(self._last_recv_t,
                                  data[self._time_label][-1])
          data.pop(self._time_label)

        # Storing the incoming data into dicts
        for label in data:
          if self._out_labels is None or label in self._out_labels:
            if label in self._buffer[link]:
              self._buffer[link][label].extend(data[label])
            else:
              self._buffer[link][label] = data[label]

    # Sending the mean value when the delay is reached
    if self._last_recv_t - self._last_sent_t > self._delay:
      ret = dict()

      # For each label of each dict, getting it mean value
      for dic in self._buffer.values():
        for label, values in dic.items():
          try:
            ret[label] = np.mean(values)
          except TypeError:
            ret[label] = values[-1]
        # Finally, clearing the buffer
        dic.clear()

      # Sending only if there was data to send
      if ret:
        ret[self._time_label] = np.mean((self._last_recv_t, self._last_sent_t))
        self.send(ret)
      self._last_sent_t = self._last_recv_t
