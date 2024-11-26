# coding: utf-8

import numpy as np
from typing import Optional
import logging

from .meta_modifier import Modifier, T


class Mean(Modifier):
  """Modifier waiting for a given number of data points to be received, then
  returning their average, and starting all over again.

  Unlike :class:`~crappy.modifier.MovingAvg`, it only returns a value once
  every ``n_points`` points.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self, n_points: int = 100) -> None:
    """Sets the args and initializes the parent class.

    Args:
      n_points: The number of points on which to compute the average.
    
        .. versionchanged:: 1.5.10 renamed from *npoints* to *n_points*
    """

    super().__init__()
    self._n_points = n_points
    self._buf = None

  def __call__(self, data: dict[str, T]) -> Optional[dict[str, T]]:
    """Receives data from the upstream Block, and computes the average of every
    label once the right number of points have been received. Then empties the
    buffer and returns the averages.

    If there are not enough points, doesn't return anything.
    
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

      # Once there's enough data in the buffer, calculating the average value
      if len(self._buf[label]) == self._n_points:
        try:
          ret[label] = float(np.mean(self._buf[label]))
        except TypeError:
          ret[label] = self._buf[label][-1]

        # Resetting the buffer
        self._buf[label].clear()

    if ret:
      self.log(logging.DEBUG, f"Sending {ret}")
      return ret

    self.log(logging.DEBUG, "Not returning any data")
