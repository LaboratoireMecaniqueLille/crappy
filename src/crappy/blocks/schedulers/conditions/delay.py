# coding: utf-8

from typing import Any
from numbers import Real

from .condition import Condition
from .... import _validation as check


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

    check.real(int(delay) if isinstance(delay, bool) else delay,
               'delay', minimum=0, strict=True)
    self._delay: float = float(delay)

  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> bool:
    """Returns :obj:`True` once ``dt`` reaches the configured delay."""

    if dt >= self._delay:
      return True
    return False
