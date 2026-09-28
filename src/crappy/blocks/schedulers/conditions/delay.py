# coding: utf-8

from typing import Any
from numbers import Real
from math import isfinite

from .condition import Condition


class Delay(Condition):
  """Becomes true after a specified time in the current State.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               delay: Real) -> None:
    """Sets the finite, strictly positive delay in seconds.

    Args:
      delay: Time to wait after entering the State.
    """

    super().__init__()

    match delay:
      case Real() if isfinite(delay) and float(delay) > 0:
        self._delay: float = float(delay)
      case Real():
        raise ValueError("The delay must be a finite, strictly positive "
                         "real number")
      case _:
        raise TypeError("The delay must be a finite, strictly positive "
                        "real number")

  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> bool:
    """Returns :obj:`True` once ``dt`` reaches the configured delay."""

    if dt >= self._delay:
      return True
    return False
