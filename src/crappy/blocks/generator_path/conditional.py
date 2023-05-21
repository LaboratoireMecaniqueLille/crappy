# coding: utf-8

from typing import Union, Dict
import logging

from .meta_path import Path, ConditionType


class Conditional(Path):
  """This Path returns one of three possible output values, based on two given
  conditions.

  It is especially useful for controlling processes that need to behave
  differently based on input values, e.g. for preventing a heating element
  from overheating, or a motor from driving too far.
  """

  def __init__(self,
               _last_time: float,
               _last_cmd: float,
               condition1: Union[str, ConditionType],
               condition2: Union[str, ConditionType],
               value1: float,
               value2: float,
               value0: float = 0) -> None:
    """Sets the args and initializes the parent class.

    Args:
      _last_time: The last timestamp when a command was generated. For internal
        use only, do not overwrite.
      _last_cmd: The last sent command. For internal use only, do not
        overwrite.
      condition1: The first condition checked by the Path. Refer to
        :class:`~crappy.blocks.generator_path.meta_path.Path` for more
        information.
      condition2: The second condition checked by the path. Refer to
        :class:`~crappy.blocks.generator_path.meta_path.Path` for more
        information.
      value1: Value to send when ``condition1`` is met.
      value2: Value to send when ``condition2`` is met and ``condition1`` is
        not met.
      value0: Value to send when neither ``condition1`` nor ``condition2`` are
        met.

    Note:
      This Generator Path never ends, it doesn't have a stop condition.
    """

    super().__init__(_last_time, _last_cmd)

    # Setting the attributes
    self._value0 = value0
    self._value1 = value1
    self._value2 = value2
    self._condition1 = self.parse_condition(condition1)
    self._condition2 = self.parse_condition(condition2)
    self._prev = self._value0

  def get_cmd(self, data: Dict[str, list]) -> float:
    """Sends either ``value1`` if ``condition1`` is met, or ``value2`` if only
    ``condition2`` is met, or ``value0`` if none of the conditions are met."""

    # Case when data has been received
    if any(data.values()):

      # Send value1 if the first condition is met
      if self._condition1(data):
        self.log(logging.DEBUG, "Condition 1 met")
        self._prev = self._value1
        return self._value1

      # Send value2 if only the second condition is met
      elif self._condition2(data):
        self.log(logging.DEBUG, "Condition 2 met")
        self._prev = self._value2
        return self._value2

      # Send value0 if no condition is met
      else:
        self.log(logging.DEBUG, "Neither condition 1 nor condition 2 met")
        self._prev = self._value0
        return self._value0

    # If no data received, return the last sent value
    else:
      return self._prev
