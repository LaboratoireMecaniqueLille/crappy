# coding: utf-8

import numpy as np
from typing import List, Optional
from time import time
import logging

from .meta_block import Block


class MeanBlock(Block):
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
               display_freq: bool = False,
               freq: Optional[float] = 50,
               debug: Optional[bool] = False) -> None:
    """Sets the args and initializes the parent class.

    Args:
      delay: The averaged data will be sent each ``delay`` seconds.
      time_label: The label containing the time information.
      out_labels: If given, only the listed labels and the time will be
        returned. Otherwise, all of them are returned.
      display_freq: If :obj:`True`, displays the looping frequency of the
        block.
      freq: The block will try to loop at this frequency.
    """

    super().__init__()
    self.display_freq = display_freq
    self.freq = freq
    self.debug = debug

    self._delay = delay
    self._time_label = time_label
    self._out_labels = out_labels

    self._last_sent_t = time()

  def begin(self) -> None:
    """Initializes the time counter."""

    self._last_sent_t = self.t0

  def loop(self) -> None:
    """Receives all available data from the upstream blocks, and averages it
    and sends it if the time delay is reached."""

    # Receiving data from each incoming link
    data = self.recv_all_data(delay=self._delay, poll_delay=self._delay / 10)
    to_send = dict()

    # Removing the time label from the received data
    if self._time_label in data:
      data.pop(self._time_label)

    # Building the output dict with the averaged values
    for label, values in data.items():
      if self._out_labels is None or label in self._out_labels:
        try:
          to_send[label] = np.mean(values)
        except (ValueError, TypeError):
          self.log(logging.WARNING, f"Cannot perform averaging on label "
                                    f"{label} with values: {values}")
          to_send[label] = values[-1]

    # Sending the output dict
    if to_send:
      to_send[self._time_label] = (time() + self._last_sent_t) / 2 - self.t0
      self._last_sent_t = time()
      self.send(to_send)
