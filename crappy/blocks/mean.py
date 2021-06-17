# coding: utf-8

import numpy as np

from .block import Block


class Mean_block(Block):
  """Can take multiple inputs, makes an average and sends the result every
  ``delay`` `s`."""

  def __init__(self, delay, tlabel='t(s)', out_labels=None, freq=50):
    """Sets the args and initializes the parent class.

    Args:
      delay (:obj:`float`): The averaged data will be sent each ``delay``
        seconds.
      tlabel (:obj:`str`, optional): The label containing the time information.
      out_labels (:obj:`list`, optional): If given, only the listed labels and
        the time will be returned. Otherwise all of them are returned.
      freq: The block will loop at this frequency.
    """

    Block.__init__(self)
    self.delay = delay
    self.tlabel = tlabel
    self.out_labels = out_labels
    self.freq = freq

  def prepare(self):
    self.temp = [dict() for _ in self.inputs]  # Will hold all the data
    self.last_t = -self.delay
    self.t = 0

  def loop(self):
    # loop over all the inputs, receive if needed, and store only
    # what we want to keep
    for i, l in enumerate(self.inputs):
      while l.poll():
        r = l.recv()
        for k in r:
          if k == self.tlabel:
            self.t = max(self.t, r[k])
          elif self.out_labels is None or k in self.out_labels:
            if k in self.temp[i]:
              self.temp[i][k].append(r[k])
            else:
              self.temp[i][k] = [r[k]]
    # If we passed delay seconds, make ;he average and send
    if self.t-self.last_t > self.delay:
      ret = {self.tlabel: (self.t + self.last_t) / 2}
      for d in self.temp:
        for k, v in d.items():
          try:
            ret[k] = np.mean(v)
          except TypeError:
            ret[k] = v[-1]
        d.clear()
      self.last_t = self.t
      self.send(ret)
