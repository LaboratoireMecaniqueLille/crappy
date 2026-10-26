# coding: utf-8

from typing import Any
from numbers import Real

from .output import Output
from .... import _validation as check


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
    
    check.string(label, 'label')
    self._label: str = label
    
    check.real(int(high) if isinstance(high, bool) else high,
               'high')
    self._high: float = float(high)

    check.real(int(low) if isinstance(low, bool) else low,
               'low')
    self._low: float = float(low)

    check.real(int(period) if isinstance(period, bool) else period,
               'period', minimum=0, strict=True)
    self._period: float = float(period)

    check.real(int(duty_cycle) if isinstance(duty_cycle, bool) else duty_cycle,
               'duty_cycle', minimum=0, maximum=1)
    self._duty_cycle: float = float(duty_cycle)
  
  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> dict[str, float]:
    """Returns ``high`` or ``low`` according to the phase of ``dt``."""
    
    ret = (self._high if dt % self._period < self._duty_cycle * self._period 
           else self._low)
    
    return {self._label: ret}
