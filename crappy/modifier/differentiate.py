# coding: utf-8

from .modifier import Modifier


class Diff(Modifier):
  """Differentiation filter.

  This will differentiate the value at ``label`` over time.

  Note:
    The time label must be specified with `time='...'`.
  """

  def __init__(self,
               label: str,
               time: str = 't(s)',
               out_label: str = None) -> None:
    Modifier.__init__(self)
    self.label = label
    self.t = time
    if out_label is None:
      self.out_label = 'd_' + self.label
    self.last_t = 0
    self.last_val = 0

  def evaluate(self, data: dict) -> dict:
    t = data[self.t]
    val = data[self.label]
    data[self.label] = (data[self.label] - self.last_val) / (t - self.last_t)
    self.last_t = t
    self.last_val = val
    return data
