# coding: utf-8

"""
Demonstration of how to create a custom out InOut in Crappy.

This InOut is intended to be used as a template, it doesn't actually act on any
real device.

No hardware required.
"""

import crappy
from time import sleep

# This class can be used as a starting point to create a new InOut object.
# To add it to crappy, make the imports relative (refer to any other inout),
# move the class to a file in crappy/inout and add the corresponding line
# in crappy/inout/__init__.py


class My_inout(crappy.inout.InOut):
  """A basic example of InOut object."""

  def __init__(self):
    # Do not forget to init InOut !
    super().__init__()

  def open(self):
    print("Opening device...")
    sleep(1)
    print("Device opened!")

  def close(self):
    print("Closing device...")
    sleep(.5)
    print("Device closed")

  def set_cmd(self, value):
    """This is the method that will be called to write the data."""

    print("Setting command to", value)


if __name__ == '__main__':
  flipflop = {
      'type': 'cyclic',
      'value1': 0,
      'condition1': 'delay=1',
      'value2': 1,
      'condition2': 'delay=1',
      'cycles': 100}

  gen = crappy.blocks.Generator([flipflop], cmd_label='cmd')

  io = crappy.blocks.IOBlock('My_inout', cmd_labels=['cmd'])

  crappy.link(gen, io)

  crappy.start()
