# coding: utf-8

from typing import Any
from numbers import Real

from .output import Output
from .... import _validation as check


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
    
    check.string(label, 'label')
    self._label: str = label

    check.real(int(amplitude) if isinstance(amplitude, bool) else amplitude,
               'amplitude', minimum=0, strict=True)
    self._amplitude: float = float(amplitude)

    check.real(int(offset) if isinstance(offset, bool) else offset,
               'offset')
    self._offset: float = float(offset)

    check.real(int(period) if isinstance(period, bool) else period,
               'period', minimum=0, strict=True)
    self._period: float = float(period)

    check.real(int(duty_cycle) if isinstance(duty_cycle, bool) else duty_cycle,
               'duty_cycle', minimum=0, maximum=1, strict=True)
    if duty_cycle == 1:
      raise ValueError("duty_cycle must be strictly less than 1")
    self._duty_cycle: float = float(duty_cycle)
  
  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> dict[str, float]:
    """Returns the triangular waveform value at the elapsed time ``dt``."""

    if (t_frac := (dt % self._period) / self._period) < self._duty_cycle:
      ret = -1 + 2 * t_frac / self._duty_cycle
    else:
      ret = 1 - 2 * (t_frac - self._duty_cycle) / (1 - self._duty_cycle)
    
    return {self._label: self._offset + self._amplitude * ret}
