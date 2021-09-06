# coding: utf-8

"""
Demonstration of how to create a custom in InOut in Crappy.

This InOut is intended to be used as a template, it doesn't actually read data
from any real device.

No hardware required.
"""

import crappy
import numpy as np
from time import time, sleep

# This class can be used as a starting point to create a new InOut object.
# To add it to crappy, make the imports relative (refer to any other inout),
# move the class to a file in crappy/inout and add the corresponding line
# in crappy/inout/__init__.py


class My_inout(crappy.inout.InOut):
  """A basic example of an InOut object."""

  def __init__(self, value, noisestd=.01):
    # Do not forget to init InOut !
    super().__init__()
    self.value = value
    self.noisestd = noisestd

  def open(self):
    print("Opening device...")
    sleep(1)
    print("Device opened!")

  def close(self):
    print("Closing device...")
    sleep(.5)
    print("Device closed")

  def get_data(self):
    """This is the method that will be called to read the data.

    It must return a list and the first value is always the time (as the number
    of seconds elapsed since 01/01/1970).
    It can return one or more other values, as long as it matches the number of
    labels in IOBlock.
    """

    return [time(), self.value + np.random.normal() * self.noisestd]


if __name__ == '__main__':
  io = crappy.blocks.IOBlock('My_inout', value=1,
                             labels=['t(s)', 'value'], freq=100)

  graph = crappy.blocks.Grapher(('t(s)', 'value'))

  crappy.link(io, graph)
  crappy.start()
