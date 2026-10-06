# coding: utf-8

from typing import Any
from numbers import Real
from math import isfinite, sin, pi

from .output import Output


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

    match frequency:
      case Real() if isfinite(frequency) and float(frequency) > 0:
        self._frequency: float = float(frequency)
      case Real():
        raise ValueError("The frequency must be a finite, strictly positive "
                         "real number")
      case _:
        raise TypeError("The frequency must be a finite, strictly positive "
                        "real number")

    match offset:
      case Real() if isfinite(offset):
        self._offset: float = float(offset)
      case Real():
        raise ValueError("The offset must be a finite real number")
      case _:
        raise TypeError("The offset must be a finite real number")

    match phase:
      case Real() if isfinite(phase):
        self._phase: float = float(phase)
      case Real():
        raise ValueError("The phase must be a finite real number")
      case _:
        raise TypeError("The phase must be a finite real number")
  
  def __call__(self,
               dt: float,
               data:  dict[str, list[Any]]) -> dict[str, float]:
    """Returns the sine-wave value at the elapsed time ``dt``."""
    
    ret = (self._offset + 
           self._amplitude * sin(2 * pi * self._frequency * dt + self._phase))
    
    return {self._label: ret}
