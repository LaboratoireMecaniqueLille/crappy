# coding: utf-8

import crappy

if __name__ == '__main__':

  gen = crappy.blocks.Generator(path=[{'type': 'Constant',
                                       'value': 5 / 60,
                                       'condition': 'delay=40'}],
                                cmd_label='input_speed')

  machine = crappy.blocks.FakeMachine(cmd_label='input_speed')

  record = crappy.blocks.Recorder(file_name='data.csv',
                                  labels=['t(s)', 'F(N)', 'x(mm)'])

  graph = crappy.blocks.Grapher(('x(mm)', 'F(N)'),
                                interp=False,
                                window_size=(6, 6))

  crappy.link(gen, machine)

  crappy.link(machine, record)
  crappy.link(machine, graph)

  crappy.start()
