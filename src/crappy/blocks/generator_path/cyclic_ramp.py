# coding: utf-8

from time import time
from typing import Union, Optional
from itertools import cycle, islice
import logging

from .meta_path import Path, ConditionType


class CyclicRamp(Path):
  """This Pth cyclically alternates between two ramps with different slopes,
  based on two different conditions.

  It is equivalent to a succession of 
  :class:`~crappy.blocks.generator_path.Ramp` Paths.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Cyclic_ramp* to *CyclicRamp*
  """

  def __init__(self,
               condition1: Union[str, ConditionType],
               condition2: Union[str, ConditionType],
               speed1: float,
               speed2: float,
               cycles: float = 1,
               init_value: Optional[float] = None) -> None:
    """Sets the arguments and initializes the parent class.

    The path always starts with ``speed1``, and then switches to ``speed2``.

    Args:
      condition1: The condition for switching to ``speed2``. Refer to
        :class:`~crappy.blocks.generator_path.meta_path.Path` for more
        information.
      condition2: The condition for switching to ``speed1``. Refer to
        :class:`~crappy.blocks.generator_path.meta_path.Path` for more
        information.
      speed1: Slope of the first generated ramp, in `units/s`.
      speed2: Slope of the second generated ramp, in `units/s`.
      cycles: Number of cycles. Half cycles are accepted. If `0`, loops
        forever.
      init_value: If given, overwrites the last value of the signal as the
        starting point for the first ramp. In the specific case when this Path
        is the first one in the Generator Paths, this argument must be given !

        .. versionadded:: 1.5.10

    Note:
      ::

        [{'type': 'CyclicRamp', 'speed1': 5, 'condition1': 'AIN0>2',
        'speed2': -2, 'condition2': 'AIN1<1', 'cycles': 5}]

      is equivalent to
      ::

        [{'type': 'Ramp', 'speed': 5,'condition': 'AIN0>2'},
        {'type': 'Ramp', 'value': -2, 'condition': 'AIN1<1'}] * 5
    
    .. versionchanged:: 1.5.10 renamed *time* argument to *_last_time*
    .. versionchanged:: 1.5.10 renamed *cmd* argument to *_last_cmd*
    .. versionremoved:: 1.5.10 *verbose* argument
    .. versionremoved:: 2.0.0 *_last_time* and *_last_cmd* arguments
    """

    super().__init__()

    if init_value is None and self.last_cmd is None:
      raise ValueError('For the first path, an init_value must be given !')

    # Creates an interator object with a given length
    if cycles > 0:
      cycles = int(2 * cycles)
      self._speeds = islice(cycle((speed1, speed2)), cycles)
      self._conditions = islice(cycle((self.parse_condition(condition1),
                                       self.parse_condition(condition2))),
                                cycles)

    # Creates an endless iterator object
    else:
      self._speeds = cycle((speed1, speed2))
      self._conditions = cycle((self.parse_condition(condition1),
                                self.parse_condition(condition2)))

    # The current condition object and value
    self._condition = None
    self._speed = None

    # The last extreme command sent
    self._last_peak_cmd = self.last_cmd if init_value is None else init_value

  def get_cmd(self, data: dict[str, list]) -> float:
    """Returns the current value of the signal and raises :exc:`StopIteration`
    when the cycles are exhausted.

    Also manages the switch between the speeds and conditions 1 and 2.
    """

    # During the first loop, getting the first condition and speed
    if self._speed is None and self._condition is None:
      try:
        self._speed = next(self._speeds)
        self._condition = next(self._conditions)
        self.log(logging.DEBUG, f"Got value {self._speed} and condition "
                                f"{self._condition}")
      except StopIteration:
        self.log(logging.DEBUG, "Stop condition met, switching to next path")
        raise

    # During other loops, getting the next condition and speed if the current
    # condition is met
    if self._condition(data):
      self.log(logging.DEBUG, "Ended phase of the cycle, switching to the "
                              "next phase")
      t = time()
      self._last_peak_cmd += self._speed * (t - self.t0)
      self.t0 = t
      try:
        self._speed = next(self._speeds)
        self._condition = next(self._conditions)
        self.log(logging.DEBUG, f"Got value {self._speed} and condition "
                                f"{self._condition}")
      except StopIteration:
        self.log(logging.DEBUG, "Stop condition met, switching to next path")
        raise

    # Finally, returning the current value
    return self._last_peak_cmd + self._speed * (time() - self.t0)
