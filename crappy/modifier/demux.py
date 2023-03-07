# coding: utf-8

import numpy as np
from typing import Dict, Any, Union, List, Tuple

from .modifier import Modifier


class Demux(Modifier):
  """Modifier converting a stream into a regular Crappy :obj:`dict` giving for
  each label a single value.

  The single value is either the first value of a column/row, or the average
  of the row/column values. This Modifier is mainly meant for linking streaming
  :ref:`IOBlock` blocks to :ref:`Grapher` blocks, as it is otherwise impossible
  to plot their data.
  """

  def __init__(self,
               labels: Union[str, List[str], Tuple[str, ...]],
               stream_label: str = "stream",
               mean: bool = False,
               time_label: str = "t(s)",
               transpose: bool = False) -> None:
    """Sets the args and initializes the parent class.

    Args:
      labels: The labels corresponding to the rows or columns of the stream.
        It can be either a single label, or a :obj:`list` of labels, or a
        :obj:`tuple` of labels. They must be given in the same order as they
        appear in the stream. If fewer labels are given than there are rows or
        columns in the stream, only the data from the first rows/columns will
        be retrieved.
      stream_label: The label carrying the stream.
      mean: If :obj:`True`, the returned value will be the average of the
        stream data. Otherwise, it will be the first value.
      time_label: The label carrying the time information.
      transpose: If :obj:`True`, each label corresponds to a row in the stream.
        Otherwise, a label corresponds to a column in the stream.
    """

    super().__init__()

    if isinstance(labels, list) or isinstance(labels, tuple):
      self._labels = labels
    else:
      self._labels = (labels,)
    self._stream_label = stream_label
    self._mean = mean
    self._time_label = time_label
    self._transpose = transpose

  def evaluate(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Retrieves for each label its value in the stream, also gets the
    corresponding timestamp, and returns them."""

    # If there are no rows or no column, cannot perform the demux
    if 0 in data[self._stream_label].shape:
      return data

    # Getting either the average or the first value for each label
    for i, label in enumerate(self._labels):
      # The data of a given label is on a same row
      if self._transpose:
        if self._mean:
          data[label] = np.mean(data[self._stream_label][i, :])
        else:
          data[label] = data[self._stream_label][i, 0]
      # The data of a given label is on a same column
      else:
        if self._mean:
          data[label] = np.mean(data[self._stream_label][:, i])
        else:
          data[label] = data[self._stream_label][0, i]

    # Discarding the raw data
    del data[self._stream_label]

    # Keeping either the average or the first time value
    if self._mean:
      data[self._time_label] = np.mean(data[self._time_label])
    else:
      data[self._time_label] = np.squeeze(data[self._time_label])[0]

    return data
