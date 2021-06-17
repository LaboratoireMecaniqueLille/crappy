# coding: utf-8

from time import time

from .path import Path


class Ramp(Path):
  """Will make a ramp from previous value until condition is reached."""

  def __init__(self, time, cmd, condition, speed):
    """Sets the args and initializes parent class.

    Args:
      time:
      cmd: If specified, will be the starting value of the ramp.
      condition (:obj:`str`): Representing the condition to end this path. See
        :ref:`generator path` for more info.
      speed: The speed of the ramp in `units/s`.
    """

    Path.__init__(self, time, cmd)
    self.condition = self.parse_condition(condition)
    self.speed = speed

  def get_cmd(self, data):
    if self.condition(data):
      raise StopIteration
    return (time() - self.t0) * self.speed + self.cmd
