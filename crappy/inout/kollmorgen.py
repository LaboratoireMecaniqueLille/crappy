from __future__ import print_function

from .inout import InOut
from ..actuator.kollmorgen import KollMorgen
from time import time

class Koll(InOut):
  def __init__(self, **kwargs):
    InOut.__init__(self)
    self.data = kwargs.pop("data", "position")
    self.motor = kwargs.pop("axis", "all")

    if self.motor == "all":
      default_label = ["t(s)"] + map(str, range(1, 4))
    else:
      default_label = ["t(s)", str(self.motor)]

    self.labels = kwargs.pop("labels", default_label)
    self.variator = KollMorgen(**kwargs)

  def open(self, **kwargs):
    pass

  def get_data(self):
    if self.data == "speed":
      if not self.motor == "all":
        ret = [time(), self.variator.read_speed(self.motor)]
      else:
        ret = [time()] + self.variator.read_speed(self.motor)

    elif self.data == "position":
      if not self.motor == "all":
        ret = [time()] + self.variator.read_position(self.motor)
      else:
        ret = [time()] + self.variator.read_position(self.motor)
    return ret

  def set_cmd(self, cmd):
    pass

  def close(self):
    pass

