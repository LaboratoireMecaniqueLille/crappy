# coding: utf-8

import numpy as np
from typing import Union

from .modifier import Modifier


class Mean(Modifier):
  """Mean filter.

  Note:
    Will divide the output `freq` by ``npoints``.

    If you need the same `freq`, see :ref:`Moving average`.

  Returns:
    The mean value every ``npoints`` point of data.
  """

  def __init__(self, npoints: int = 100) -> None:
    """Sets the instance attributes.

    Args:
      npoints (:obj:`int`): The number of points it takes to return `1` value.
    """

    Modifier.__init__(self)
    self.npoints = npoints

  def evaluate(self, data: dict) -> Union[dict, None]:
    if not hasattr(self, "last"):
      self.last = dict(data)
      for k in data:
        self.last[k] = [self.last[k]]
      return data
    r = {}
    for k in data:
      self.last[k].append(data[k])
      if len(self.last[k]) == self.npoints:
        try:
          r[k] = np.mean(self.last[k])
        except TypeError:  # Non numeric data
          r[k] = self.last[k][-1]
      elif len(self.last[k]) > self.npoints:
        self.last[k] = []
    if r:
      return r
