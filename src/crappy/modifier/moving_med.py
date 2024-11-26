# coding: utf-8

import numpy as np
import logging

from .meta_modifier import Modifier, T


class MovingMed(Modifier):
  """Modifier replacing the data of each label with its median value over a
  chosen number of points.

  Unlike :class:`~crappy.modifier.Median`, it returns a value each time data is
  received from the upstream Block.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Moving_med* to *MovingMed*
  """

  def __init__(self, n_points: int = 100) -> None:
    """Sets the args and initializes the parent class.

    Args:
      n_points: The maximum number of points on which to compute the median.
    
        .. versionchanged:: 1.5.10 renamed from *npoints* to *n_points*
    """

    super().__init__()
    self._n_points = n_points
    self._buf = None

  def __call__(self, data: dict[str, T]) -> dict[str, T]:
    """Receives data from the upstream Block, computes the median of every
    label and replaces the original data with it.
    
    .. versionchanged:: 2.0.0 renamed from *evaluate* to *__call__*
    """

    self.log(logging.DEBUG, f"Received {data}")

    # Initializing the buffer
    if self._buf is None:
      self._buf = {key: [value] for key, value in data.items()}

    ret = {}
    for label in data:
      # Updating the buffer with the newest data
      self._buf[label].append(data[label])

      # Trimming the buffer if there's too much data
      if len(self._buf[label]) > self._n_points:
        self._buf[label] = self._buf[label][-self._n_points:]

      # Calculating the median for each label
      try:
        ret[label] = float(np.median(self._buf[label]))
      except TypeError:
        ret[label] = self._buf[label][-1]

    self.log(logging.DEBUG, f"Sending {ret}")
    return ret
