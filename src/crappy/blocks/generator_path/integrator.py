# coding: utf-8

from numpy import trapz
from typing import Union
import logging

from .meta_path import Path, ConditionType


class Integrator(Path):
  """This Path integrates an incoming label over time and returns the
  integration as an output signal.

  Let `f(t)` be the input signal, `v(t)` the value of the output, `m` the
  inertia and `t0` the timestamp of the beginning of this Path.

  Then the output value for this Path will be
  :math:`v(t) = v(t0) - [I(t0 -> t)f(t)dt] / m`.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Inertia* to *Integrator*
  """

  def __init__(self,
               condition: Union[str, ConditionType],
               inertia: float,
               func_label: str,
               time_label: str = 't(s)',
               init_value: float = None) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      condition: The condition for switching to the next Path. Refer to
        :class:`~crappy.blocks.generator_path.meta_path.Path` for more
        information.
      inertia: Value of the equivalent inertia to use for driving the signal.
        In the above formula, it is the value of `m`. The larger this value,
        the slower the changes in the signal value.
      func_label: The name of the label of the input value to integrate.

        .. versionchanged:: 1.5.10 renamed from *flabel* to *func_label*
      time_label: The name of the time label for the integration.

        .. versionchanged:: 1.5.10 renamed from *tlabel* to *time_label*
      init_value: If given, overwrites the last value of the signal as the
        starting point for the inertia path. In the specific case when this
        path is the first one in the Generator Paths, this argument must be
        given !

        .. versionchanged:: 1.5.10 renamed from *value* to *init_value*
    
    .. versionchanged:: 1.5.10 renamed *time* argument to *_last_time*
    .. versionchanged:: 1.5.10 renamed *cmd* argument to *_last_cmd*
    .. versionremoved:: 1.5.10 *const* argument
    .. versionremoved:: 2.0.0 *_last_time* and *_last_cmd* arguments
    """

    super().__init__()

    if init_value is None and self.last_cmd is None:
      raise ValueError('For the first path, an init_value must be given !')

    # Setting the attributes
    self._condition = self.parse_condition(condition)
    self._time_label = time_label
    self._func_label = func_label
    self._inertia = inertia
    self._value = self.last_cmd if init_value is None else init_value
    self._last_t = None
    self._last_val = None

  def get_cmd(self, data: dict[str, list]) -> float:
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
