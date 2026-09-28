# coding: utf-8

from collections.abc import Callable
from typing import Any

from .condition import Condition


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

    match conditions:
      case ():
        raise ValueError("At least one condition must be provided")
      case (*conditions,) if all(isinstance(cond, Condition) or callable(cond)
                                 for cond in conditions):
        self._conditions: tuple[
          Condition | Callable[[float,  dict[str, list[Any]]],
                               bool], ...] = tuple(conditions)
      case _:
        raise TypeError("conditions must be instances of Condition or "
                        "callables")

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
