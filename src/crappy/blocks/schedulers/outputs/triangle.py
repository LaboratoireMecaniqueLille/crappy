# coding: utf-8

from typing import Any
from numbers import Real
from math import isfinite

from .output import Output


class Triangle(Output):
  """Outputs a periodic triangular waveform for one label.

  Each period starts at ``offset - amplitude``, rises to
  ``offset + amplitude`` at the duty-cycle point, and falls back to its
  starting value.

  .. versionadded:: 2.1.0
  """
  
  def __init__(self, 
               label: str,
               amplitude: Real,
               period: Real,
               offset: Real = 0.0,
               duty_cycle: Real = 0.5) -> None:
    """Sets the waveform amplitude, period, offset and duty cycle.

    Args:
      label: The non-empty label under which to send the value.
      amplitude: Half the peak-to-peak range, must be finite and positive.
      period: The duration of one cycle in seconds, must be finite and
        positive.
      offset: The midpoint of the waveform, must be finite.
      duty_cycle: The fraction of each period spent rising. Must be strictly
        between `0` and `1`.
    """
    
    super().__init__()
    
    match label:
      case str() if label.strip():
        self._label: str = label
      case str():
        raise ValueError("The provided label must be a non-empty string")
      case _:
        raise TypeError("The provided label must be a non-empty string")

    match amplitude:
      case Real() if isfinite(amplitude) and float(amplitude) > 0:
        self._amplitude: float = float(amplitude)
      case Real():
        raise ValueError("The amplitude must be a finite, strictly positive "
                         "real number")
      case _:
        raise TypeError("The amplitude must be a finite, strictly positive "
                        "real number")

    match offset:
      case Real() if isfinite(offset):
        self._offset: float = float(offset)
      case Real():
        raise ValueError("The offset must be a finite real number")
      case _:
        raise TypeError("The offset must be a finite real number")

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
      case Real() if isfinite(duty_cycle) and 0 < float(duty_cycle) < 1:
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
    """Returns the triangular waveform value at the elapsed time ``dt``."""

    if (t_frac := (dt % self._period) / self._period) < self._duty_cycle:
      ret = -1 + 2 * t_frac / self._duty_cycle
    else:
      ret = 1 - 2 * (t_frac - self._duty_cycle) / (1 - self._duty_cycle)
    
    return {self._label: self._offset + self._amplitude * ret}
