# coding: utf-8

from typing import Optional
import logging

from .meta_modifier import Modifier, T


class DownSampler(Modifier):
  """Modifier waiting for a given number of data points to be received, then
  returning only the last received point.

  Similar to :class:`~crappy.modifier.Mean`, except it discards the values 
  that are not transmitted instead of averaging them. Useful for reducing
  the amount of data sent to a Block.
  
  .. versionadded:: 2.0.4
  """

  def __init__(self, n_points: int = 10) -> None:
    """Sets the args and initializes the parent class.

    Args:
      n_points: One value will be sent to the downstream Block only once
        every ``n_points`` received values.
    """

    super().__init__()
    self._n_points: int = n_points
    self._count: int = n_points - 1

  def __call__(self, data: dict[str, T]) -> Optional[dict[str, T]]:
    """Receives data from the upstream Block, and if the counter matches the 
    threshold, returns the data.

    If the counter doesn't match the threshold, doesn't return anything and 
    increments the counter.
    """

    self.log(logging.DEBUG, f"Received {data}")

    if self._count == self._n_points - 1:
      self._count = 0
      self.log(logging.DEBUG, f"Sending {data}")
      return data
    
    else:
      self._count += 1
      self.log(logging.DEBUG, "Not returning any data")
