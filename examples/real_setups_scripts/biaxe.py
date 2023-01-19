# coding: utf-8

"""
This program is used to control a biaxial tensile machine.

It demonstrates an equibiaxial cyclic test.
"""

import crappy

if __name__ == '__main__':

  # The instructions for driving the motors
  path1 = {'type': 'Cyclic', 'value1': 5, 'condition1': 'delay=3',
           'value2': -5, 'condition2': 'delay=3', 'cycles': 0}
  path2 = {'type': 'Cyclic', 'value1': 5, 'condition1': 'delay=5',
           'value2': -5, 'condition2': 'delay=5', 'cycles': 0}

  # The signal generators driving the motors
  gen_1 = crappy.blocks.Generator(path=[path1], cmd_label="vx")
  gen_2 = crappy.blocks.Generator(path=[path2], cmd_label="vy")

  # Settings common to all four motors to drive
  mot = {'type': 'Biaxe', 'mode': 'speed'}
  # Settings specific to each motor
  mot_a = {'port': '/dev/ttyS4', 'cmd': 'vy'}
  mot_b = {'port': '/dev/ttyS5', 'cmd': 'vx'}
  mot_c = {'port': '/dev/ttyS6', 'cmd': 'vy'}
  mot_d = {'port': '/dev/ttyS7', 'cmd': 'vx'}

  # Instantiating the Block driving the motors and linking it to the Generator
  machine = crappy.blocks.Machine([mot_a, mot_b, mot_c, mot_d], common=mot)
  crappy.link(gen_1, machine)
  crappy.link(gen_2, machine)

  # Starting the test
  crappy.start()
