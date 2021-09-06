# coding: utf-8

"""
Example demonstrating the use of ``crappy.blocks.Multiplex``.

This block interpolates data from several parents in a common timebase.

No required hardware.
"""

import crappy


class Delay:
  """Modifier to add a delay to one of the inputs, demonstrating how Multiplex
  will wait for data."""

  def __init__(self, n):
    self.n = n
    self.hist = []

  def evaluate(self, data):
    self.hist.append(data)
    if len(self.hist) >= self.n:
      return self.hist.pop(0)


if __name__ == "__main__":
  g1 = crappy.blocks.Generator([
    dict(type='sine', freq=1, amplitude=1, condition=None)
      ], freq=100, cmd_label='cmd1')

  g2 = crappy.blocks.Generator([
    dict(type='cyclic_ramp', speed2=-1, speed1=1,
         condition1='cmd2>1', condition2='cmd2<-1', cycles=1e30)
      ], freq=50, cmd_label='cmd2')

  mul = crappy.blocks.Multiplex()

  # crappy.link(g1, mul)
  crappy.link(g1, mul, modifier=Delay(50))
  crappy.link(g2, mul)

  graph = crappy.blocks.Grapher(('t(s)', 'cmd1'), ('t(s)', 'cmd2'))

  crappy.link(mul, graph)

  rec = crappy.blocks.Recorder("example_multi.csv",
                               labels=["t(s)", "cmd1", "cmd2"])

  crappy.link(mul, rec)

  crappy.start()
