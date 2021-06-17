# coding: utf-8

from time import time
import numpy as np

from .path import Path


class Custom(Path):
  """To generate a custom path from a file."""

  def __init__(self, time, cmd, filename, delimiter='\t'):
    """Loads the file and sets the args.

    Args:
      time:
      cmd:
      filename: Name of the `.csv` file.

        Note:
          It must contain two columns: one with time, the other with the value.

      delimiter:
    """

    Path.__init__(self, time, cmd)
    with open(filename, 'r') as f:
      self.array = np.loadtxt(f, delimiter=delimiter)
    assert len(self.array.shape) == 2 and 2 in self.array.shape,\
        "Custom array must have shape (N,2) or (2,N)"
    if self.array.shape[1] == 2:
      self.t = self.array[:, 0]
      self.f = self.array[:, 1]
    else:
      self.t = self.array[0, :]
      self.f = self.array[1, :]

  def get_cmd(self, data):
    t = time()
    if t - self.t0 > max(self.t):
      raise StopIteration
    return np.interp(t - self.t0, self.t, self.f)
