# coding: utf-8

from typing import Any
from numbers import Real
from math import isfinite

from .output import Output


class Ramp(Output):
  """Outputs a value that changes linearly with time in the State.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               label: str,
               init_value: Real,
               slope: Real) -> None:
    """Sets the starting value and the rate of change.

    Args:
      label: The non-empty label under which to send the value.
      init_value: The finite value at the moment the State is entered.
      slope: The finite change in value per second, may be negative.
    """

    super().__init__()

    match label:
      case str() if label.strip():
        self._label: str = label
      case str():
        raise ValueError("The provided label must be a non-empty string")
      case _:
        raise TypeError("The provided label must be a non-empty string")

    match init_value:
      case Real() if isfinite(init_value):
        self._init_value: float = float(init_value)
      case Real():
        raise ValueError("The init_value must be a finite real number")
      case _:
        raise TypeError("The init_value must be a finite real number")
    
    match slope:
      case Real() if isfinite(slope):
        self._slope: float = float(slope)
      case Real():
        raise ValueError("The slope must be a finite real number")
      case _:
        raise TypeError("The slope must be a finite real number")

  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> dict[str, float]:
    """Returns ``init_value + dt * slope`` under the output label."""

    return {self._label: self._init_value + dt * self._slope}
