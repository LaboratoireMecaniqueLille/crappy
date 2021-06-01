# coding: utf-8

import numpy as np

from .modifier import Modifier


class Median(Modifier):
  """
  Median filter.

  Returns:
    The median value every npoints point of data.

  Arg:
    - npoints (int): The number of points it takes to return 1 value.

  """

  def __init__(self, npoints=100):
    Modifier.__init__(self)
    self.npoints = npoints

  def evaluate(self, data):
    if not hasattr(self, "last"):
      self.last = dict(data)
      for k in self.last:
        self.last[k] = [self.last[k]]
      return data
    r = {}
    for k in data:
      self.last[k].append(data[k])
      if len(self.last[k]) == self.npoints:
        try:
          r[k] = np.median(self.last[k])
        except TypeError:
          r[k] = self.last[k][-1]
      elif len(self.last[k]) > self.npoints:
        self.last[k] = []
    if r:
      return r
