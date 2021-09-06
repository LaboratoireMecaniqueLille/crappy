# coding: utf-8

"""
Demonstration of how to create a custom Block in Crappy.

This block is intended to be used as a template, it doesn't actually act on
any device nor do anything exceptional.

No hardware required.
"""

import crappy
from time import time

# This class can be used as a starting point to create a new block. All methods
# except loop are optional.
# To add it to crappy, make the imports relative (refer to any other block),
# move the class to a file in crappy/blocks and add the corresponding line
# in crappy/blocks/__init__.py


# crappy.Block is a shortcut for crappy.blocks.Block
class MyBlock(crappy.Block):
  """A basic example of block.

  It will simply send a sawtooth signal.
  """

  def __init__(self, period=1):  # Optional
    # If you define your own constructor, do not forget to init Block !
    super().__init__()
    # Example argument, here the period of the sawtooth
    self.period = period

    # These attributes can be specified, they will be interpreted by Block
    self.freq = 100  # Hz
    self.labels = ['t(s)', 'cmd(V)']

  def prepare(self):  # Optional
    # Called inside the Process, before the actual beginning of the test
    # Create the objects needed for the test here
    # (ie. open a camera, a serial port etc...)
    print("==> prepare has been called")

  def begin(self):  # Optional
    # Now the test has begun, thi method will be called once
    print("==> begin has been called")
    # Send -1 only on the first loop
    # self.t0 contains the timestamp of the beginning of the test
    self.send([time() - self.t0, -1])

  def loop(self):
    # This loop will be called continuously until the end of the test
    # print("==> loop")
    t = time() - self.t0
    # Because self.labels is defined, we can send a list of the same
    # length, each value will be assigned to a label
    self.send([t, t % self.period])
    # But we could have written:
    # self.send({'t(s)': t, 'cmd(V)': t % self.period})

  def finish(self):  # Optional
    # This will be called before ending the process
    print("==> finish has been called")


if __name__ == "__main__":
  mb = MyBlock()
  # NB: matplotlib must be installed to used the Grapher
  graph = crappy.blocks.Grapher(('t(s)', 'cmd(V)'))
  crappy.link(mb, graph)

  crappy.start()
