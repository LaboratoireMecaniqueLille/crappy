# coding: utf-8

import crappy

if __name__ == '__main__':

  gen = crappy.blocks.Generator(path=[{'type': 'Cyclic',
                                       'value1': 5/60,
                                       'value2': -5/60,
                                       'condition1': 'delay=10',
                                       'condition2': 'delay=10',
                                       'cycles': 2},
                                      {'type': 'Constant',
                                       'value': 5 / 60,
                                       'condition': 'delay=40'}],
                                cmd_label='input_speed')

  machine = crappy.blocks.FakeMachine(cmd_label='input_speed')

  record = crappy.blocks.Recorder(file_name='data.csv',
                                  labels=['t(s)', 'F(N)', 'x(mm)'])

  graph_force = crappy.blocks.Grapher(('t(s)', 'F(N)'))

  graph_pos = crappy.blocks.Grapher(('t(s)', 'x(mm)'))

  crappy.link(gen, machine)

  crappy.link(machine, record)
  crappy.link(machine, graph_pos)
  crappy.link(machine, graph_force)

  crappy.start()
