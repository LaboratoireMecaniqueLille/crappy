# coding: utf-8

from typing import Union, Dict
from warnings import warn
from .path import Path, condition_type


class Constant(Path):
  """The simplest path, simply sends the same value until the condition is
  met."""

  def __init__(self,
               _last_time: float,
               _last_cmd: float,
               condition: Union[str, condition_type],
               value: float = None) -> None:
    """Sets the args and initializes the parent class.

    Args:
      _last_time: The last timestamp when a command was generated. For internal
        use only, do not overwrite.
      _last_cmd: The last sent command. For internal use only, do not
        overwrite.
      condition: The condition for switching to the next path. Refer to
        :ref:`generator path` for more info.
      value: The value to send.
    """

    warn("The _last_time and _last_cmd arguments will be removed in version "
         "2.0.0", DeprecationWarning)

    Path.__init__(self, _last_time, _last_cmd)

    self._condition = self.parse_condition(condition)

    if value is None and _last_cmd is None:
      raise ValueError('For the first path, a value must be given !')

    self._value = _last_cmd if value is None else value

  def get_cmd(self, data: Dict[str, list]) -> float:
    """Returns the value to send or raises :exc:`StopIteration` if the stop
    condition is met."""

    # Checking if the stop condition is met
    if self._condition(data):
      raise StopIteration

    # Returning the value
    return self._value
