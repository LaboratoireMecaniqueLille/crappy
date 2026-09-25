# coding: utf-8

import numpy as np
from collections.abc import Sequence
from time import time
import logging
from numbers import Real
from math import isfinite
from collections import defaultdict
from typing import Any

from .meta_block import Block


class MeanBlock(Block):
  """This Block computes the average values of given labels over successive
  time windows.

  It can take any number of inputs. Incoming values are continuously buffered,
  then averaged and sent at the end of each window. If a configured label
  cannot be averaged, its latest value is sent instead. If no selected label
  was received during a window, nothing is sent for that window.

  Incoming Blocks should normally share a common time label. When time values
  are available, the output timestamp is the midpoint of their range. When
  they are not, it is the midpoint of the averaging window. If the same label
  other than time is received from several Blocks, it may lead to unexpected
  results.
  
  The output of this Block is very similar to that of the 
  :class:`~crappy.modifier.Mean` and :class:`~crappy.modifier.MovingAvg` 
  Modifiers, but not exactly similar. While these Modifiers calculate the 
  average of a label over a fixed number of data points, the MeanBlock 
  calculates the average of the values received over a given delay. This 
  behavior could, however, also be achieved using a 
  :class:`~crappy.modifier.meta_modifier.modifier.Modifier`.

  Warning:
    If the delay for averaging is too short compared with the looping frequency
    of the upstream Blocks, this Block may not always return the same number of
    labels ! This can cause errors in downstream Blocks expecting a fixed
    number of labels.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Mean_block* to *MeanBlock*
  .. versionchanged:: 2.1.0 buffer data without blocking between averaging
     windows
  """

  def __init__(self,
               delay: float,
               time_label: str = 't(s)',
               out_labels: str | Sequence[str] | None = None,
               display_freq: bool = False,
               freq: float | None = 50,
               debug: bool | None = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      delay: Duration of each averaging window, in seconds, as a finite,
        strictly positive number. At the end of each window, the buffered data
        is averaged and sent if at least one selected label was received.
      time_label: The non-empty label containing the time information. It
        should be common to all incoming Links. When no value is received under
        this label, the output timestamp is calculated from the averaging
        window.
        
        .. versionchanged:: 1.5.10 renamed from *t_label* to *time_label*
      out_labels: A non-empty sequence (like a :obj:`list` or a :obj:`tuple`)
        containing the non-empty labels to average, as :obj:`str`. If not
        given, all received labels except the time label are processed. The
        time label should not be included, as it is already given in
        ``time_label``. If there is only one label to average, it can be given
        directly as a :obj:`str`, i.e. not in a sequence.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
        
        .. versionadded:: 1.5.10
        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      freq: The target looping frequency for the Block. This determines how
        often incoming Links are drained into the averaging buffer, it does
        not change the duration configured by ``delay``. If :obj:`None`, loops
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

    match delay:
      case Real() if not isfinite(delay) or delay <= 0:
        raise ValueError("delay must be a finite strictly positive float")
      case Real():
        self._delay: float = delay
      case _:
        raise TypeError("delay must be a finite strictly positive float")
    match time_label:
      case str() if time_label:
        self._time_label: str = time_label
      case str():
        raise ValueError("time_label must be a non-empty string")
      case _:
        raise TypeError("time_label must be a non-empty string")
    match out_labels:
      case None:
        self._out_labels: list[str] | None = out_labels
      case str() if out_labels:
        self._out_labels: list[str] | None = [out_labels]
      case str():
        raise ValueError("out_labels must be a non-empty string or a sequence "
                         "of non-empty strings or None")
      case ():
        raise ValueError("out_labels must contain at least one element when "
                         "provided as a sequence")
      case (*out_labels,) if (all(isinstance(label, str) for
                                  label in out_labels) and
                              all(label for label in out_labels)):
        self._out_labels: list[str] | None = list(out_labels)
      case (*_,):
        raise ValueError("out_labels must be a non-empty string or a sequence "
                         "of non-empty strings or None")
      case _:
        raise TypeError("out_labels must be a non-empty string or a sequence "
                        "of non-empty strings or None")

    self._last_sent_t: float = time()
    self._data_buf: dict[str, list[Any]] = defaultdict(list)

  def prepare(self) -> None:
    """Checks that there's at least one incoming and one output
    :class:`~crappy.links.link.Link`.

    .. versionadded:: 2.0.9
    """

    if not self.inputs:
      raise IOError("No Link pointing towards the Mean Block!")
    if not self.outputs:
      raise IOError("The Mean Block has no output Link!")

  def begin(self) -> None:
    """Initializes the start time of the first averaging window.
    
    .. versionadded:: 2.0.0
    """

    self._last_sent_t = time()

  def loop(self) -> None:
    """Buffers all available input and processes completed averaging windows.

    Once ``delay`` seconds have elapsed, the selected buffered values are
    averaged and sent, and the buffer is cleared for the next window. A window
    containing no selected values is cleared without sending data.

    .. versionchanged:: 2.0.9 average time label computed directly from time
      data when possible
    .. versionchanged:: 2.1.0 receive data without blocking and buffer it
      between loop iterations
    """

    time_data: list[float] | None = None

    # Receiving data from each incoming link
    data = self.recv_all_data()

    # Store the received data in the averaging buffer
    for label, values in data.items():
      self._data_buf[label].extend(values)

    # If the averaging window is not complete yet, just return
    if time() - self._last_sent_t < self._delay:
      self.log(logging.DEBUG, "Not yet time to average data, returning")
      return

    to_send = dict()

    # Remove the time label from the buffered data
    if self._time_label in self._data_buf:
      time_data = self._data_buf.pop(self._time_label)

    # Building the output dict with the averaged values
    for label, values in self._data_buf.items():
      if self._out_labels is None or label in self._out_labels:
        try:
          to_send[label] = float(np.mean(values))
        except (ValueError, TypeError):
          self.log(logging.WARNING, f"Cannot perform averaging on label "
                                    f"{label} with values: {values}")
          to_send[label] = values[-1]

    # Clear the buffer
    self._data_buf.clear()

    # Sending the output dict
    if to_send:
      if time_data is not None:
        to_send[self._time_label] = (min(time_data) + max(time_data)) / 2
      else:
        to_send[self._time_label] = (time() + self._last_sent_t) / 2 - self.t0
      self.send(to_send)
    else:
      self.log(logging.DEBUG, "No data to send for this loop, although delay "
                              "is passed")

    # Start the next averaging window even if no data was sent
    self._last_sent_t = time()
