# coding: utf-8

from collections.abc import Callable
from typing import Any

from .condition import Condition
from .... import _validation as check


class AnyCondition(Condition):
  """Combines conditions with logical OR.

  Conditions are checked in the order provided. Evaluation stops at the
  first true result.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               *conditions: Condition | Callable[[float,
                                                  dict[str, list[Any]]], bool]
               ) -> None:
    """Stores the conditions to combine.

    Args:
      *conditions: One or more :class:`Condition` objects or callables
        accepting elapsed State time and received data and returning a
        truthy or false value.
    """

    super().__init__()

    check.sequence(conditions, 'conditions', minimum=1)
    for condition in conditions:
      check.callable_(condition, 'conditions')
    self._conditions: tuple[
      Condition | Callable[[float, dict[str, list[Any]]], bool],
      ...] = tuple(conditions)

  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> bool:
    """Returns :obj:`True` if at least one condition is true."""

    return any(cond(dt, data) for cond in self._conditions)

  def reset(self) -> None:
    """Calls ``reset()`` on nested conditions that provide it."""

    for condition in self._conditions:
      if hasattr(condition, 'reset') and callable(condition.reset):
        condition.reset()
