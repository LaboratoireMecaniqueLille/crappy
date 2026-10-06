# coding: utf-8

from typing import Any
from numbers import Real
from math import isfinite

from .output import Output


class Square(Output):
  """Outputs a periodic two-level waveform for one label.

  .. versionadded:: 2.1.0
  """
  
  def __init__(self, 
               label: str,
               high: Real,
               low: Real,
               period: Real,
               duty_cycle: Real = 0.5) -> None:
    """Sets the two levels, period and duty cycle.

    Args:
      label: The non-empty label under which to send the value.
      high: The finite value output during the first part of each period.
      low: The finite value output during the rest of each period.
      period: The duration of one cycle in seconds, must be finite and
        positive.
      duty_cycle: The fraction of each period spent at ``high``. May range
        from `0` (always ``low``) to `1` (always ``high``).
    """
    
    super().__init__()
    
    match label:
      case str() if label.strip():
        self._label: str = label
      case str():
        raise ValueError("The provided label must be a non-empty string")
      case _:
        raise TypeError("The provided label must be a non-empty string")
    
    match high:
      case Real() if isfinite(high):
        self._high: float = float(high)
      case Real():
        raise ValueError("The high value must be a finite real number")
      case _:
        raise TypeError("The high value must be a finite real number")

    match low:
      case Real() if isfinite(low):
        self._low: float = float(low)
      case Real():
        raise ValueError("The low value must be a finite real number")
      case _:
        raise TypeError("The low value must be a finite real number")

    match period:
      case Real() if isfinite(period) and float(period) > 0:
        self._period: float = float(period)
      case Real():
        raise ValueError("The period must be a finite, strictly positive "
                         "real number")
      case _:
        raise TypeError("The period must be a finite, strictly positive "
                        "real number")

    match duty_cycle:
      case Real() if isfinite(duty_cycle) and 0 <= float(duty_cycle) <= 1:
        self._duty_cycle: float = float(duty_cycle)
      case Real():
        raise ValueError("The duty_cycle must be a finite real number between "
                         "0 and 1")
      case _:
        raise TypeError("The duty_cycle must be a finite real number between "
                        "0 and 1")
  
  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> dict[str, float]:
    """Returns ``high`` or ``low`` according to the phase of ``dt``."""
    
    ret = (self._high if dt % self._period < self._duty_cycle * self._period 
           else self._low)
    
    return {self._label: ret}
