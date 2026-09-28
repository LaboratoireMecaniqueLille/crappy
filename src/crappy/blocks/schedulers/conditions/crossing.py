# coding: utf-8

from typing import Any, Literal
from numbers import Real
from math import isfinite
import logging
import operator

from .condition import Condition


class Crossing(Condition):
  """Detects values crossing a threshold in the requested direction.

  A rising crossing requires a value strictly below the threshold followed
  by one strictly above it. A falling crossing reverses these comparisons.
  The first side observation is retained across calls until a crossing is
  detected or :meth:`reset` is called. Values for the watched label should
  come from one incoming Link, since values from several Links are grouped by
  Link rather than merged in chronological order.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               label: str,
               threshold: Real,
               direction: Literal['rising', 'falling']) -> None:
    """Sets the watched label, threshold and crossing direction.

    Args:
      label: The non-empty input label to inspect.
      threshold: The finite threshold value.
      direction: ``'rising'`` for below-to-above crossings, or ``'falling'``
        for above-to-below crossings.
    """

    super().__init__()

    match label:
      case str() if label.strip():
        self._label: str = label
      case str():
        raise ValueError("The provided label must be a non-empty string")
      case _:
        raise TypeError("The provided label must be a non-empty string")

    match threshold:
      case Real() if isfinite(threshold):
        self._threshold: float = float(threshold)
      case Real():
        raise ValueError("The threshold must be a finite real number")
      case _:
        raise TypeError("The threshold must be a finite real number")

    match direction:
      case str() if direction in ('rising', 'falling'):
        self._direction: Literal['rising', 'falling'] = direction
      case str():
        raise ValueError("direction must be either 'rising' or 'falling'")
      case _:
        raise TypeError("direction must be either 'rising' or 'falling'")

    # Whether there was once a value read on the right side of the threshold
    # for detecting a crossing (e.g. True if reading value 5 for threshold 6
    # in 'rising' mode, but False if only read 7)
    self._prev_on_right_side: bool | None = None

  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> bool:
    """Checks the received values in order for the selected crossing.

    Clears the earlier side observation after a crossing so that a later
    reading on the same side does not report the crossing again.
    """

    # First check that the condition can be evaluated
    if self._label not in data:
      self.log(logging.DEBUG, f"label {self._label} missing from the input "
                              f"data, cannot check the condition")
      return False

    prev_op = operator.lt if self._direction == 'rising' else operator.gt
    thresh_op = operator.gt if self._direction == 'rising' else operator.lt

    # Cannot detect a crossing if we never detect a value on the right side
    # of the threshold
    values = iter(data[self._label])
    while self._prev_on_right_side is None or not self._prev_on_right_side:
      try:
        self._prev_on_right_side = prev_op(next(values), self._threshold)
      except StopIteration:
        return False

    # Once there is a value detected on the right side of the threshold, we
    # can check for values crossing the threshold
    while True:
      try:
        if thresh_op(next(values), self._threshold):
          self._prev_on_right_side = False
          return True
      except StopIteration:
        return False

  def reset(self) -> None:
    """Forgets the earlier observation on State entry."""

    self.log(logging.DEBUG, "Crossing Condition was reset")
    self._prev_on_right_side = None
