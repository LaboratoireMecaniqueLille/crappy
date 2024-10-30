# coding: utf-8

from time import time
from numpy import sin, pi
from typing import Union
import logging

from .meta_path import Path, ConditionType


class Sine(Path):
  """This Path generates a sine wave varying with time until the stop condition
  is met.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               condition: Union[str, ConditionType],
               freq: float,
               amplitude: float,
               offset: float = 0,
               phase: float = 0) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      condition: The condition for switching to the next Path. Refer to
        :class:`~crappy.blocks.generator_path.meta_path.Path` for more
        information.
      freq: The frequency of the sine in `Hz`.
      amplitude: The amplitude of the sine wave (peak to peak).
      offset: The offset of the sine (average value).
      phase: The phase of the sine (in radians).
    
    .. versionchanged:: 1.5.10 renamed *time* argument to *_last_time*
    .. versionchanged:: 1.5.10 renamed *cmd* argument to *_last_cmd*
    .. versionremoved:: 2.0.0 *_last_time* and *_last_cmd* arguments
    """

    super().__init__()

    # Setting the attributes
    self._condition = self.parse_condition(condition)
    self._amplitude = amplitude / 2
    self._offset = offset
    self._phase = phase
    self._k = 2 * pi * freq

  def get_cmd(self, data: dict[str, list]) -> float:
    """Returns the value to send or raises :exc:`StopIteration` if the stop
    condition is met."""

    # Checking if the stop condition is met
    if self._condition(data):
      self.log(logging.DEBUG, "Stop condition met")
      raise StopIteration

    # Returning the current signal value
    return (sin((time() - self.t0) * self._k - self._phase) *
            self._amplitude + self._offset)
