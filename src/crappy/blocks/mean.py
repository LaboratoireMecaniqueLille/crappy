# coding: utf-8

import numpy as np
from typing import Optional, Union
from collections.abc import Iterable
from time import time
import logging

from .meta_block import Block


class MeanBlock(Block):
  """This Block can compute the average values of given labels over a given 
  delay.

  It can take any number of inputs, provided that they share a common time
  label. If the same label (except time) is received from several Blocks, it
  may lead to unexpected results.
  
  The output of this Block is very similar to that of the 
  :class:`~crappy.modifier.Mean` and :class:`~crappy.modifier.MovingAvg` 
  Modifiers, but not exactly similar. While these Modifiers calculate the 
  average of a label over a fixed number of data points, the MeanBlock 
  calculates the average of the values received over a given delay. This 
  behavior could, however, also be achieved using a 
  :class:`~crappy.modifier.Modifier`.

  Warning:
    If the delay for averaging is too short compared with the looping frequency
    of the upstream Blocks, this Block may not always return the same number of
    labels ! This can cause errors in downstream Blocks expecting a fixed
    number of labels.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Mean_block* to *MeanBlock*
  """

  def __init__(self,
               delay: float,
               time_label: str = 't(s)',
               out_labels: Optional[Union[str, Iterable[str]]] = None,
               display_freq: bool = False,
               freq: Optional[float] = 50,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      delay: The averaged data will be sent each ``delay`` seconds.
      time_label: The label containing the time information. It must be common
        to all the incoming Links.
        
        .. versionchanged:: 1.5.10 renamed from *t_label* to *time_label*
      out_labels: An iterable (like a :obj:`list` or a :obj:`tuple`) containing
        all the labels to average, as :obj:`str`. If not given, all the
        received labels are averaged and returned. The time label should not
        be included, as it is already given in ``time_label``. If there is only
        one label to average, it can be directly given as a :obj:`str`, i.e.
        not in an iterable.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
        
        .. versionadded:: 1.5.10
        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      freq: The target looping frequency for the Block. If :obj:`None`, loops 
        as fast as possible.
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
        
        .. versionadded:: 2.0.0
    """

    super().__init__()
    self.display_freq = display_freq
    self.freq = freq
    self.debug = debug

    self._delay = delay
    self._time_label = time_label

    # Forcing the out_labels into a list
    if out_labels is not None and isinstance(out_labels, str):
      self._out_labels = [out_labels]
    elif out_labels is not None:
      self._out_labels = list(out_labels)
    else:
      self._out_labels = None

    self._last_sent_t = time()

  def begin(self) -> None:
    """Initializes the time counter.
    
    .. versionadded:: 2.0.0
    """

    self._last_sent_t = self.t0

  def loop(self) -> None:
    """Receives all available data from the upstream Blocks, averages it and
    sends it if the time delay is reached."""

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
          to_send[label] = float(np.mean(values))
        except (ValueError, TypeError):
          self.log(logging.WARNING, f"Cannot perform averaging on label "
                                    f"{label} with values: {values}")
          to_send[label] = values[-1]

    # Sending the output dict
    if to_send:
      to_send[self._time_label] = (time() + self._last_sent_t) / 2 - self.t0
      self._last_sent_t = time()
      self.send(to_send)
