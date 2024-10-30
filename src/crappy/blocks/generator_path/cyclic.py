# coding: utf-8

from time import time
from typing import Union
from itertools import cycle, islice
import logging

from .meta_path import Path, ConditionType


class Cyclic(Path):
  """This Path cyclically alternates between two constant values, based on two
  different conditions.

  It can for example be used as a trigger, or used to drive an actuator
  cyclically. It is equivalent to a succession of 
  :class:`~crappy.blocks.generator_path.Constant` Paths.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               condition1: Union[str, ConditionType],
               condition2: Union[str, ConditionType],
               value1: float,
               value2: float,
               cycles: float = 1) -> None:
    """Sets the arguments and initializes the parent class.

    The Path always starts with ``value1``, and then switches to ``value2``.

    Args:
      condition1: The condition for switching to ``value2``. Refer to
        :class:`~crappy.blocks.generator_path.meta_path.Path` for more
        information.
      condition2: The condition for switching to ``value1``. Refer to
        :class:`~crappy.blocks.generator_path.meta_path.Path` for more
        information.
      value1: First value to send.
      value2: Second value to send.
      cycles: Number of cycles. Half cycles are accepted. If `0`, loops
        forever.

    Note:
      ::

        [{'type': 'Cyclic', 'value1': 1, 'condition1': 'AIN0>2',
        'value2': 0, 'condition2': 'AIN1<1', 'cycles': 5}]

      is equivalent to
      ::

        [{'type': 'Constant', 'value': 1,'condition': 'AIN0>2'},
        {'type': 'Constant', 'value': 0, 'condition': 'AIN1<1'}] * 5
    
    .. versionchanged:: 1.5.10 renamed *time* argument to *_last_time*
    .. versionchanged:: 1.5.10 renamed *cmd* argument to *_last_cmd*
    .. versionremoved:: 1.5.10 *verbose* argument
    .. versionremoved:: 2.0.0 *_last_time* and *_last_cmd* arguments
    """

    super().__init__()

    # Creates an interator object with a given length
    if cycles > 0:
      cycles = int(2 * cycles)
      self._values = islice(cycle((value1, value2)), cycles)
      self._conditions = islice(cycle((self.parse_condition(condition1),
                                       self.parse_condition(condition2))),
                                cycles)

    # Creates an endless iterator object
    else:
      self._values = cycle((value1, value2))
      self._conditions = cycle((self.parse_condition(condition1),
                                self.parse_condition(condition2)))

    # The current condition object and value
    self._condition = None
    self._value = None

  def get_cmd(self, data: dict[str, list]) -> float:
    """Returns either the first or second value depending on the current state
    of the cycle. Raises :exc:`StopIteration` when the cycles are exhausted.

    Also manages the switch between the values and conditions 1 and 2.
    """

    # During the first loop, getting the first condition and value
    if self._value is None and self._condition is None:
      try:
        self._value = next(self._values)
        self._condition = next(self._conditions)
        self.log(logging.DEBUG, f"Got value {self._value} and condition "
                                f"{self._condition}")
      except StopIteration:
        self.log(logging.DEBUG, "No value or condition, switching to next "
                                "path")
        raise

    # During other loops, getting the next condition and value if the current
    # condition is met
    if self._condition(data):
      self.log(logging.DEBUG, "Ended phase of the cycle, switching to the "
                              "next phase")
      self.t0 = time()
      try:
        self._value = next(self._values)
        self._condition = next(self._conditions)
        self.log(logging.DEBUG, f"Got value {self._value} and condition "
                                f"{self._condition}")
      except StopIteration:
        self.log(logging.DEBUG, "Stop condition met, switching to next path")
        raise

    # Finally, returning the current value
    return self._value
