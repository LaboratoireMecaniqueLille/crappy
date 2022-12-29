# coding: utf-8

from time import time
from typing import Union, Dict, Optional
from itertools import cycle, islice

from .path import Path, Condition_type


class Cyclic_ramp(Path):
  """The path cyclically alternates between two ramps with different slopes,
  based on two different conditions.

  It is equivalent to a succession of :ref:`ramp` paths.
  """

  def __init__(self,
               _last_time: float,
               _last_cmd: float,
               condition1: Union[str, Condition_type],
               condition2: Union[str, Condition_type],
               speed1: float,
               speed2: float,
               cycles: float = 1,
               init_value: Optional[float] = None) -> None:
    """Sets the args and initializes the parent class.

    The path always starts with ``speed1``, and then switches to ``speed2``.

    Args:
      _last_time: The last timestamp when a command was generated. For internal
        use only, do not overwrite.
      _last_cmd: The last sent command. For internal use only, do not
        overwrite.
      condition1: The condition for switching to ``speed2``. Refer to
        :ref:`generator path` for more info.
      condition2: The condition for switching to ``speed1``. Refer to
        :ref:`generator path` for more info.
      speed1: Slope of the first generated ramp, in `units/s`.
      speed2: Slope of the second generated ramp, in `units/s`.
      cycles: Number of cycles. Half cycles are accepted. If `0`, loops
        forever.
      init_value: If given, overwrites the last value of the signal as the
        starting point for the first ramp. In the specific case when this path
        is the first one in the list of dicts, this argument must be given !

    Note:
      ::

        [{'type': 'cyclic_ramp', 'speed1': 5, 'condition1': 'AIN0>2',
        'speed2': -2, 'condition2': 'AIN1<1', 'cycles': 5}]

      is equivalent to
      ::

        [{'type': 'ramp', 'speed': 5,'condition': 'AIN0>2'},
        {'type': 'ramp', 'value': -2, 'condition': 'AIN1<1'}] * 5
    """

    Path.__init__(self, _last_time, _last_cmd)

    if init_value is None and _last_cmd is None:
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
    self._last_peak_cmd = _last_cmd if init_value is None else init_value

  def get_cmd(self, data: Dict[str, list]) -> float:
    """Returns the current value of the signal and raises :exc:`StopIteration`
    when the cycles are exhausted.

    Also manages the switch between the speeds and conditions 1 and 2.
    """

    # During the first loop, getting the first condition and speed
    if self._speed is None and self._condition is None:
      try:
        self._speed = next(self._speeds)
        self._condition = next(self._conditions)
      except StopIteration:
        raise

    # During other loops, getting the next condition and speed if the current
    # condition is met
    if self._condition(data):
      t = time()
      self._last_peak_cmd += self._speed * (t - self.t0)
      self.t0 = t
      try:
        self._speed = next(self._speeds)
        self._condition = next(self._conditions)
      except StopIteration:
        raise

    # Finally, returning the current value
    return self._last_peak_cmd + self._speed * (time() - self.t0)
