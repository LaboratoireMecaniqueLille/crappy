# coding: utf-8

import numpy as np
from typing import Any, Union
from collections.abc import Iterable
import logging

from .meta_modifier import Modifier


class Demux(Modifier):
  """Modifier converting a stream into a regular data flow interpretable by
  most Blocks.

  It is meant to be used on a :class:`~crappy.links.Link` taking an
  :class:`~crappy.blocks.IOBlock` in streamer mode as an input. It converts
  the stream to make it readable by most Blocks, and also splits the stream in
  several labels if necessary.

  It takes a stream as an input, i.e. a :obj:`dict` whose values are
  :obj:`numpy.array`, and outputs another :obj:`dict` whose values are
  :obj:`float`. If the numpy arrays contains several columns (corresponding to
  several acquired channels), it splits them into several labels.

  Important:
    In the process of converting the stream data to regular labeled data, much
    information is lost ! This Modifier is intended to format the stream data
    for low-frequency plotting, or low-frequency decision-making. To save all
    the stream data, use the :class:`~crappy.blocks.HDFRecorder` Block.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               labels: Union[str, Iterable[str]],
               stream_label: str = "stream",
               mean: bool = False,
               time_label: str = "t(s)",
               transpose: bool = False) -> None:
    """Sets the args and initializes the parent class.

    Args:
      labels: The labels corresponding to the rows or columns of the stream.
        It can be either a single label, or an iterable of labels (like a
        :obj:`list` or a :obj:`tuple`). They must be given in the same order as
        they appear in the stream. If fewer labels are given than there are
        rows or columns in the stream, only the data from the first rows or
        columns will be retrieved.
      stream_label: The label carrying the stream.

        .. versionchanged:: 1.5.10 renamed from *stream* to *stream_label*
      mean: If :obj:`True`, the returned value will be the average of the
        row or column. Otherwise, it will be the first value.
      time_label: The label carrying the time information.
      transpose: If :obj:`True`, each label corresponds to a row in the stream.
        Otherwise, a label corresponds to a column in the stream.
    """

    super().__init__()

    if isinstance(labels, str):
      labels = (labels,)
    self._labels = labels

    self._stream_label = stream_label
    self._mean = mean
    self._time_label = time_label
    self._transpose = transpose

  def __call__(self, data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Retrieves for each label its value in the stream, also gets the
    corresponding timestamp, and returns them.
    
    .. versionchanged:: 1.5.10 
       merge *evaluate_mean* and *evaluate_nomean* methods into *evaluate*
    .. versionchanged:: 2.0.0 renamed from *evaluate* to *__call__*
    """

    self.log(logging.DEBUG, f"Received {data}")

    # If there are no rows or no column, cannot perform the demux
    if 0 in data[self._stream_label].shape:
      return data

    # Getting either the average or the first value for each label
    for i, label in enumerate(self._labels):
      # The data of a given label is on a same row
      if self._transpose:
        if self._mean:
          data[label] = float(np.mean(data[self._stream_label][i, :]))
        else:
          data[label] = float(data[self._stream_label][i, 0])
      # The data of a given label is on a same column
      else:
        if self._mean:
          data[label] = float(np.mean(data[self._stream_label][:, i]))
        else:
          data[label] = float(data[self._stream_label][0, i])

    # Discarding the raw data
    del data[self._stream_label]

    # Keeping either the average or the first time value
    if self._mean:
      data[self._time_label] = float(np.mean(data[self._time_label]))
    else:
      data[self._time_label] = float(np.squeeze(data[self._time_label])[0])

    self.log(logging.DEBUG, f"Sending {data}")

    return data
