# coding: utf-8

from time import time
from typing import Callable, Union, Dict, Optional
from re import split, IGNORECASE, match
import logging
from multiprocessing import current_process

ConditionType = Callable[[Dict[str, list]], bool]


class Path:
  """Parent class for all the generator paths.

  Allows them to have access to the :meth:`parse_condition` method.
  """

  def __init__(self,
               _last_time: float,
               _last_cmd: Optional[float] = None) -> None:
    """Simply sets the arguments."""

    self.t0 = _last_time
    self.last_cmd = _last_cmd if _last_cmd is not None else 0
    self._logger: Optional[logging.Logger] = None

  def get_cmd(self, _: Dict[str, list]) -> float:
    """If not overridden, simply returns the last_cmd attribute."""

    return self.last_cmd

  def log(self, level: int, msg: str) -> None:
    """"""

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)

  def parse_condition(
        self,
        condition: Optional[Union[str, ConditionType]]) -> ConditionType:
    """This method returns a function allowing to check whether the stop
    condition is met or not.

    Its main use is to parse the conditions given as strings, but it can also
    accept :obj:`None` or a callable as arguments.

    If given as a string, the supported condition types are :
    ::

      '<var> > <threshold>'
      '<var> < <threshold>'
      'delay = <your_delay>'

    With ``<var>``, ``<threshold>`` and ``<your_delay>`` to be replaced
    respectively with the label on which the condition applies, the threshold
    for the condition to become true, and the delay before switching to the
    next path.
    """

    if not isinstance(condition, str):
      # First case, the condition is None
      if condition is None:
        self.log(logging.DEBUG, "Condition is None")
        return lambda _: False
      # Second case, the condition is already a Callable
      elif isinstance(condition, Callable):
        self.log(logging.DEBUG, "Condition is a callable")
        return condition

    # Third case, the condition is a string containing '<'
    if '<' in condition:
      self.log(logging.DEBUG, "Condition is of type var < thresh")
      var, thresh = split(r'\s*<\s*', condition)

      # Return a function that checks if received data is inferior to threshold
      def cond(data: Dict[str, list]) -> bool:
        if var in data:
          return any((val < float(thresh) for val in data[var]))
        return False

      return cond

    # Fourth case, the condition is a string containing '>'
    elif '>' in condition:
      self.log(logging.DEBUG, "Condition is of type var > thresh")
      var, thresh = split(r'\s*>\s*', condition)

      # Return a function that checks if received data is superior to threshold
      def cond(data: Dict[str, list]) -> bool:
        if var in data:
          return any((val > float(thresh) for val in data[var]))
        return False

      return cond

    # Fifth case, it is a delay condition
    elif match(r'delay', condition, IGNORECASE) is not None:
      self.log(logging.DEBUG, "Condition is of type delay=xx")
      delay = float(split(r'=\s*', condition)[1])
      # Return a function that checks if the delay is expired
      return lambda _: time() - self.t0 > delay

    # Otherwise, it's an invalid syntax
    else:
      raise ValueError("Wrong syntax for the condition, please refer to the "
                       "documentation")
