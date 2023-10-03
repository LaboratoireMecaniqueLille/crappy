# coding: utf-8

import crappy

if __name__ == '__main__':

  gen = crappy.blocks.Generator(({'type': 'Cyclic',
                                  'value1': -10,
                                  'condition1': 'delay=3',
                                  'value2': 10,
                                  'condition2': 'delay=3',
                                  'cycles': 3},),
                                freq=50,
                                cmd_label='tension(V)')

  mot = crappy.blocks.Machine(({'type': 'FakeDCMotor',
                                'cmd_label': 'tension(V)',
                                'mode': 'speed',
                                'speed_label': 'speed(RPM)',
                                'kv': 500},),
                              freq=50)

  graph = crappy.blocks.Grapher(('t(s)', 'speed(RPM)'))

  crappy.link(gen, mot)
  crappy.link(mot, graph)

  crappy.start()
