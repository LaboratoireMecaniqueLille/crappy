# coding: utf-8

from time import time
from typing import Union, Dict, Optional
from warnings import warn

from .path import Path, condition_type


class Ramp(Path):
  """Sends a ramp signal varying linearly over time, until the stop condition
  is met."""

  def __init__(self,
               _last_time: float,
               _last_cmd: float,
               condition: Union[str, condition_type],
               speed: float,
               init_value: Optional[float] = None):
    """Sets the args and initializes the parent class.

    Args:
      _last_time: The last timestamp when a command was generated. For internal
        use only, do not overwrite.
      _last_cmd: The last sent command. For internal use only, do not
        overwrite.
      condition: The condition for switching to the next path. Refer to
        :ref:`generator path` for more info.
      speed: The slope of the ramp, in `units/s`.
      init_value: If given, overwrites the last value of the signal as the
        starting point for the ramp. In the specific case when this path is the
        first one in the list of dicts, this argument must be given !
    """

    warn("The _last_time and _last_cmd arguments will be removed in version "
         "2.0.0", DeprecationWarning)

    Path.__init__(self, _last_time, _last_cmd)

    if init_value is None and _last_cmd is None:
      raise ValueError('For the first path, an init_value must be given !')

    # Setting the attributes
    self._condition = self.parse_condition(condition)
    self._speed = speed
    self._init_value = init_value if init_value is not None else _last_cmd

  def get_cmd(self, data: Dict[str, list]) -> float:
    """Returns the value to send or raises :exc:`StopIteration` if the stop
    condition is met."""

    # Checking if the stop condition is met
    if self._condition(data):
      raise StopIteration

    # Returning the current value
    return self._init_value + (time() - self.t0) * self._speed
