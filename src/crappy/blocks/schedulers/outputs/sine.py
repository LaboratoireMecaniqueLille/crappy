# coding: utf-8

from typing import Any
from numbers import Real
from math import sin, pi

from .output import Output
from .... import _validation as check


class Sine(Output):
  """Outputs a sine wave for one label.

  .. versionadded:: 2.1.0
  """
  
  def __init__(self, 
               label: str,
               amplitude: Real,
               frequency: Real,
               offset: Real,
               phase: Real = 0) -> None:
    """Sets the waveform amplitude, frequency, offset and phase.

    Args:
      label: The non-empty label under which to send the value.
      amplitude: Half the peak-to-peak range, must be finite and positive.
      frequency: The number of cycles per second, must be finite and positive.
      offset: The midpoint of the waveform, must be finite.
      phase: The initial phase in radians, must be finite.
    """
    
    super().__init__()
    
    check.string(label, 'label')
    self._label: str = label
    
    check.real(int(amplitude) if isinstance(amplitude, bool) else amplitude,
               'amplitude', minimum=0, strict=True)
    self._amplitude: float = float(amplitude)

    check.real(int(frequency) if isinstance(frequency, bool) else frequency,
               'frequency', minimum=0, strict=True)
    self._frequency: float = float(frequency)

    check.real(int(offset) if isinstance(offset, bool) else offset,
               'offset')
    self._offset: float = float(offset)

    check.real(int(phase) if isinstance(phase, bool) else phase,
               'phase')
    self._phase: float = float(phase)
  
  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> dict[str, float]:
    """Returns the sine-wave value at the elapsed time ``dt``."""
    
    ret = (self._offset + 
           self._amplitude * sin(2 * pi * self._frequency * dt + self._phase))
    
    return {self._label: ret}
