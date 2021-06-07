# coding: utf-8

from .modifier import Modifier


class Integrate(Modifier):
  """Integration filter.

  This will integrate the value at ``label`` over time.

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
      self.out_label = 'i_' + self.label
    else:
      self.out_label = out_label
    self.last_t = 0
    self.val = 0

  def evaluate(self, data: dict) -> dict:
    t = data[self.t]
    self.val += (t - self.last_t) * data[self.label]
    self.last_t = t
    data[self.out_label] = self.val
    return data
