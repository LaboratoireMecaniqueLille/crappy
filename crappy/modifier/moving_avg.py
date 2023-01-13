# coding: utf-8

import numpy as np
from typing import Dict, Any
import logging

from .meta_modifier import Modifier


class Moving_avg(Modifier):
  """Modifier replacing the data of each label with its average value over a
  chosen number of points.

  Unlike :ref:`Mean`, it returns a value each time a message is received from
  the upstream link.
  """

  def __init__(self, n_points: int = 100) -> None:
    """Sets the args and initializes the parent class.

    Args:
      n_points: The maximum number of points on which to compute the average.
    """

    super().__init__()
    self._n_points = n_points
    self._buf = None

  def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Receives data from the upstream block, computes the average of every
    label and replaces the original data with it."""

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

      # Calculating the average for each label
      try:
        ret[label] = np.mean(self._buf[label])
      except TypeError:
        ret[label] = self._buf[label][-1]

    self.log(logging.DEBUG, f"Sending {ret}")
    return ret
