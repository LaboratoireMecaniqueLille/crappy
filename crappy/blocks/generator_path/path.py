# coding: utf-8

from time import time
from typing import Callable, Union


class Path(object):
  """Parent class for all paths."""

  def __init__(self, time: float, cmd: float) -> None:
    self.t0 = time
    self.cmd = cmd

  def get_cmd(self, data: dict) -> float:
    return self.cmd

  def parse_condition(self, condition: Union[str, bool, Callable]) -> Callable:
    """This method turns a string into a function that returns a bool.

    It is meant to check if a skip condition is reached.

    The following syntax is supported:
      - ``myvar>myvalue``
      - ``myvar<myvalue``
      - ``delay=mydelay``

    Note:
      `myvar` must be the label of an input value, `myvalue` should be a
      :obj:`float`. This will return :obj:`True` when the data under the label
      `myvar` is larger/smaller than `myvalue`.

      The condition will turn :obj:`True` after `mydelay` seconds.

      Any other syntax will return :obj:`True`.
    """

    if not isinstance(condition, str):
      if condition is None or not condition:
        return lambda _: False  # For never ending conditions
      elif isinstance(condition, Callable):
        return condition
    if '<' in condition:
      var, val = condition.split('<')
      return lambda data: any([i < float(val) for i in data[var]])
    elif '>' in condition:
      var, val = condition.split('>')
      return lambda data: any([i > float(val) for i in data[var]])
    elif condition.startswith('delay'):
      val = float(condition.split('=')[1])
      return lambda data: time() - self.t0 > val
    else:
      raise ValueError("Wrong syntax for the condition, please refer to the "
                       "documentation")
