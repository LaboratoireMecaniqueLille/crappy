# coding: utf-8

from numpy import trapz
from typing import Union, Dict
import logging

from .meta_path import Path, ConditionType


class Integrator(Path):
  """This Path integrates an incoming label over time and returns the
  integration as an output signal.

  Let `f(t)` be the input signal, `v(t)` the value of the output, `m` the
  inertia and `t0` the timestamp of the beginning of this Path.

  Then the output value for this Path will be
  :math:`v(t) = v(t0) - [I(t0 -> t)f(t)dt] / m`.
  """

  def __init__(self,
               _last_time: float,
               _last_cmd: float,
               condition: Union[str, ConditionType],
               inertia: float,
               func_label: str,
               time_label: str = 't(s)',
               init_value: float = None) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      _last_time: The last timestamp when a command was generated. For internal
        use only, do not overwrite.
      _last_cmd: The last sent command. For internal use only, do not
        overwrite.
      condition: The condition for switching to the next Path. Refer to
        :class:`~crappy.blocks.generator_path.meta_path.Path` for more
        information.
      inertia: Value of the equivalent inertia to use for driving the signal.
        In the above formula, it is the value of `m`. The larger this value,
        the slower the changes in the signal value.
      func_label: The name of the label of the input value to integrate.
      time_label: The name of the time label for the integration.
      init_value: If given, overwrites the last value of the signal as the
        starting point for the inertia path. In the specific case when this
        path is the first one in the Generator Paths, this argument must be
        given !
    """

    super().__init__(_last_time, _last_cmd)

    if init_value is None and _last_cmd is None:
      raise ValueError('For the first path, an init_value must be given !')

    # Setting the attributes
    self._condition = self.parse_condition(condition)
    self._time_label = time_label
    self._func_label = func_label
    self._inertia = inertia
    self._value = _last_cmd if init_value is None else init_value
    self._last_t = None
    self._last_val = None

  def get_cmd(self, data: Dict[str, list]) -> float:
    """Gets the latest values of the incoming label, integrates them and
    changes the output accordingly.

    Also raises :exc:`StopIteration` in case the stop condition is met.
    """

    # Checking if the stop condition is met
    if self._condition(data):
      self.log(logging.DEBUG, "Stop condition met")
      raise StopIteration

    if self._time_label in data and self._func_label in data:
      # Getting the last values from the received data
      times = data[self._time_label]
      values = data[self._func_label]

      # Including the last values from the last loop
      if self._last_val is not None and self._last_t is not None:
        times = [self._last_t] + times
        values = [self._last_val] + values

      # Keeping in memory the last values from this loop
      if times and values:
        self._last_t = times[-1]
        self._last_val = values[-1]

      # Performing the integration and subtracting from the previous value
      self._value -= trapz(values, times) / self._inertia

    # Returning the current value
    return self._value
