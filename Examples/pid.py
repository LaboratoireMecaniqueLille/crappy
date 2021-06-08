# coding: utf-8

"""
Quick example showing how the PID block can be used to control the speed of a
motor.

The motor is a virtual entity, taking a voltage and returning a speed and
position.

No hardware required.
"""

import crappy

if __name__ == "__main__":

  g = crappy.blocks.Generator([
    {'type': 'constant', 'value': 1000, 'condition': 'delay=3'},
    {'type': 'ramp', 'speed': 100, 'condition': 'delay=5', 'cmd': 0},
    {'type': 'constant', 'value': 1800, 'condition': 'delay=3'},
    {'type': 'constant', 'value': 500, 'condition': 'delay=3'},
    {'type': 'sine', 'amplitude': 2000, 'offset': 1000, 'freq': .3,
      'condition': 'delay=15'}
  ], spam=True)

  kv = 1000

  mot = crappy.blocks.Machine([{'type': 'Fake_motor',
                               'cmd': 'pid',
                               'mode': 'speed',
                               'speed_label': 'speed',
                               # Motor properties:
                               'kv': kv,
                               'inertia': 4,
                               'rv': .2,
                               'fv': 1e-5
                                }])
  graph_m = crappy.blocks.Grapher(('t(s)', 'speed'), ('t(s)', 'cmd'))
  # , interp=False)

  crappy.link(mot, graph_m)
  crappy.link(g, graph_m)
  # To see what happens without PID
  # crappy.link(g, mot)
  # crappy.start()

  p = 38 / kv
  i = 2
  d = .05

  pid = crappy.blocks.PID(kp=p,
                          ki=i,
                          kd=d,
                          out_max=10,
                          out_min=-10,
                          i_limit=.5,
                          input_label='speed',
                          send_terms=True)

  crappy.link(g, pid)
  crappy.link(pid, mot)
  # crappy.link(mot, pid)  # This line will not smooth the feedback
  crappy.link(mot, pid, modifier=crappy.modifier.Moving_avg(15))

  graph_pid = crappy.blocks.Grapher(('t(s)', 'pid'))
  crappy.link(pid, graph_pid)

  graph_pid2 = crappy.blocks.Grapher(('t(s)', 'p_term'),
                                     ('t(s)', 'i_term'),
                                     ('t(s)', 'd_term'))

  crappy.link(pid, graph_pid2)

  crappy.start()
