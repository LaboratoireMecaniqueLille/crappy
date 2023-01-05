# coding: utf-8

from time import time
from typing import Union, Dict
from itertools import cycle, islice
import logging

from .path import Path, Condition_type


class Cyclic(Path):
  """The path cyclically alternates between two constant values, based on two
  different conditions.

  It can for example be used as a trigger, or used to drive an actuator
  cyclically. It is equivalent to a succession of :ref:`constant` paths.
  """

  def __init__(self,
               _last_time: float,
               _last_cmd: float,
               condition1: Union[str, Condition_type],
               condition2: Union[str, Condition_type],
               value1: float,
               value2: float,
               cycles: float = 1) -> None:
    """Sets the args and initializes the parent class.

    The path always starts with ``value1``, and then switches to ``value2``.

    Args:
      _last_time: The last timestamp when a command was generated. For internal
        use only, do not overwrite.
      _last_cmd: The last sent command. For internal use only, do not
        overwrite.
      condition1: The condition for switching to ``value2``. Refer to
        :ref:`generator path` for more info.
      condition2: The condition for switching to ``value1``. Refer to
        :ref:`generator path` for more info.
      value1: First value to send.
      value2: Second value to send.
      cycles: Number of cycles. Half cycles are accepted. If `0`, loops
        forever.

    Note:
      ::

        [{'type': 'cyclic', 'value1': 1, 'condition1': 'AIN0>2',
        'value2': 0, 'condition2': 'AIN1<1', 'cycles': 5}]

      is equivalent to
      ::

        [{'type': 'constant', 'value': 1,'condition': 'AIN0>2'},
        {'type': 'constant', 'value': 0, 'condition': 'AIN1<1'}] * 5
    """

    super().__init__(_last_time, _last_cmd)

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

  def get_cmd(self, data: Dict[str, list]) -> float:
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
