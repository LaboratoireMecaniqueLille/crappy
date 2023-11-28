# coding: utf-8

from time import time
from typing import Union, Dict, Optional
import logging

from .meta_path import Path, ConditionType


class Ramp(Path):
  """Sends a ramp signal varying linearly over time, until the stop condition
  is met."""

  def __init__(self,
               condition: Union[str, ConditionType],
               speed: float,
               init_value: Optional[float] = None):
    """Sets the arguments and initializes the parent class.

    Args:
      condition: The condition for switching to the next Path. Refer to
        :class:`~crappy.blocks.generator_path.meta_path.Path` for more
        information.
      speed: The slope of the ramp, in `units/s`.
      init_value: If given, overwrites the last value of the signal as the
        starting point for the ramp. In the specific case when this path is the
        first one in the Generator Paths, this argument must be given !
    """

    super().__init__()

    if init_value is None and self.last_cmd is None:
      raise ValueError('For the first path, an init_value must be given !')

    # Setting the attributes
    self._condition = self.parse_condition(condition)
    self._speed = speed
    self._init_value = init_value if init_value is not None else self.last_cmd

  def get_cmd(self, data: Dict[str, list]) -> float:
    """Returns the value to send or raises :exc:`StopIteration` if the stop
    condition is met."""

    # Checking if the stop condition is met
    if self._condition(data):
      self.log(logging.DEBUG, "Stop condition met")
      raise StopIteration

    # Returning the current value
    return self._init_value + (time() - self.t0) * self._speed
