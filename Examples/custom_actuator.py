# coding: utf-8

"""
Demonstration of how to create a custom Actuator in Crappy.

This actuator is intended to be used as a template, it doesn't actually act on
any device.

No hardware required.
"""

import crappy
from time import sleep

# This class can be used as a starting point to create a new Actuator object
# To add it to crappy, make the imports relative (refer to any other
# actuators),
# move the class to a file in crappy/actuator and add the corresponding line
# in crappy/actuator/__init__.py


class My_actuator(crappy.actuator.Actuator):
  """A basic example of an Actuator object."""

  def __init__(self):
    # Do not forget to init Actuator !
    super().__init__()

  def open(self):
    print("Opening device...")
    sleep(1)
    self.pos = 0
    print("Device opened!")

  def close(self):
    print("Closing device...")
    sleep(.5)
    print("Device closed")

  # At least one of the two following methods (set_speed and set_position) needs
  # to be defined. Getters (get_speed and get_position) can be defined too if
  # the actuator supports it.
  def set_position(self, target):
    self.pos = target

  def get_position(self):
    return self.pos + 0.1  # To illustrate the difference with the target

  def stop(self):
    """Called before closing, should stop the actuator."""

    print("Stopping the actuator")


if __name__ == '__main__':
  sine_path = {'type': 'sine',
               'amplitude': 1,
               'freq': 1,
               'condition': None}

  gen = crappy.blocks.Generator([sine_path],
                                cmd_label='target_position',)

  machine = crappy.blocks.Machine([{
      'type': 'My_actuator',  # The class to instantiate
      'mode': 'position',  # set_position will be called
      'cmd': 'target_position',  # This actuator will be driven using sine_label
      'pos_label': 'measured_position'}])  # The label to send the measurements

  crappy.link(gen, machine)

  graph = crappy.blocks.Grapher(('t(s)', 'measured_position'),
                                ('t(s)', 'target_position'))

  crappy.link(machine, graph)
  crappy.link(gen, graph)

  crappy.start()
