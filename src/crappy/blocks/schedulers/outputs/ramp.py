# coding: utf-8

from typing import Any
from numbers import Real

from .output import Output
from .... import _validation as check


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

    check.string(label, 'label')
    self._label: str = label

    check.real(int(init_value) if isinstance(init_value, bool) else init_value,
               'init_value')
    self._init_value: float = float(init_value)
    
    check.real(int(slope) if isinstance(slope, bool) else slope,
               'slope')
    self._slope: float = float(slope)

  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> dict[str, float]:
    """Returns ``init_value + dt * slope`` under the output label."""

    return {self._label: self._init_value + dt * self._slope}
