# coding: utf-8

import numpy as np
from typing import Dict, Any, Optional
import logging

from .meta_modifier import Modifier


class Median(Modifier):
  """Modifier waiting for a given number of data points to be received, then
  returning their median, and starting all over again.

  Unlike :class:`~crappy.modifier.MovingMed`, it only returns a value once
  every ``n_points`` points.
  """

  def __init__(self, n_points: int = 100) -> None:
    """Sets the args and initializes the parent class.

    Args:
      n_points: The number of points on which to compute the median.
    """

    super().__init__()
    self._n_points = n_points
    self._buf = None

  def __call__(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Receives data from the upstream Block, and computes the median of every
    label once the right number of points have been received. Then empties the
    buffer and returns the medians.

    If there are not enough points, doesn't return anything.
    """

    self.log(logging.DEBUG, f"Received {data}")

    # Initializing the buffer
    if self._buf is None:
      self._buf = {key: [value] for key, value in data.items()}

    ret = {}
    for label in data:
      # Updating the buffer with the newest data
      self._buf[label].append(data[label])

      # Once there's enough data in the buffer, calculating the median value
      if len(self._buf[label]) == self._n_points:
        try:
          ret[label] = np.median(self._buf[label])
        except TypeError:
          ret[label] = self._buf[label][-1]

        # Resetting the buffer
        self._buf[label].clear()

    if ret:
      self.log(logging.DEBUG, f"Sending {ret}")
      return ret

    self.log(logging.DEBUG, "Not returning any data")
