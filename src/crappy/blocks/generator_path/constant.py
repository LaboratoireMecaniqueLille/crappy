# coding: utf-8

from typing import Union
import logging

from .meta_path import Path, ConditionType


class Constant(Path):
  """The simplest Path, outputs the same constant value until the stop 
  condition is met.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               condition: Union[str, ConditionType],
               value: float = None) -> None:
    """Sets the args and initializes the parent class.

    Args:
      condition: The condition for switching to the next Path. Refer to
        :class:`~crappy.blocks.generator_path.meta_path.Path` for more
        information.
      value: The value to output.
      
    .. versionchanged:: 1.5.10 renamed *time* argument to *_last_time*
    .. versionchanged:: 1.5.10 renamed *cmd* argument to *_last_cmd*
    .. versionremoved:: 1.5.10 *send_one* argument
    .. versionremoved:: 2.0.0 *_last_time* and *_last_cmd* arguments
    """

    super().__init__()

    self._condition = self.parse_condition(condition)

    if value is None and self.last_cmd is None:
      raise ValueError('For the first path, a value must be given !')

    self._value = self.last_cmd if value is None else value

  def get_cmd(self, data: dict[str, list]) -> float:
    """Returns the value to send or raises :exc:`StopIteration` if the stop
    condition is met.
    
    .. versionadded:: 1.5.10
    """

    # Checking if the stop condition is met
    if self._condition(data):
      self.log(logging.DEBUG, "Stop condition met")
      raise StopIteration

    # Returning the value
    return self._value
