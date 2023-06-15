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
    {'type': 'Constant', 'value': 1000, 'condition': 'delay=3'},
    {'type': 'Ramp', 'speed': 100, 'condition': 'delay=5', 'init_value': 0},
    {'type': 'Constant', 'value': 1800, 'condition': 'delay=3'},
    {'type': 'Constant', 'value': 500, 'condition': 'delay=3'},
    {'type': 'Sine', 'amplitude': 2000, 'offset': 1000, 'freq': .3,
     'condition': 'delay=15'}
  ], spam=True)

  kv = 1000

  mot = crappy.blocks.Machine([{'type': 'FakeDCMotor',
                                'cmd_label': 'pid',
                                'mode': 'speed',
                                'speed_label': 'speed',
                                # Motor properties:
                                'kv': kv,
                                'inertia': 4,
                                'rv': .2,
                                'fv': 1e-5}])
  graph_m = crappy.blocks.Grapher(('t(s)', 'speed'), ('t(s)', 'cmd'))
  # , interp=False)

  crappy.link(mot, graph_m, modifier=crappy.modifier.Mean(10))
  crappy.link(g, graph_m, modifier=crappy.modifier.Mean(10))
  # To see what happens without PID
  # crappy.link(g, mot)
  # crappy.start()

  p = 38 / kv
  i = 76 / kv
  d = 1.9 / kv

  pid = crappy.blocks.PID(kp=p,
                          ki=i,
                          kd=d,
                          out_max=10,
                          out_min=-10,
                          i_limit=(-5, 5),
                          input_label='speed',
                          send_terms=True)

  crappy.link(g, pid)
  crappy.link(pid, mot)
  # crappy.link(mot, pid)  # This line will not smooth the feedback
  crappy.link(mot, pid, modifier=crappy.modifier.MovingAvg(15))

  graph_pid = crappy.blocks.Grapher(('t(s)', 'pid'))
  crappy.link(pid, graph_pid, modifier=crappy.modifier.Mean(10))

  graph_pid2 = crappy.blocks.Grapher(('t(s)', 'p_term'), ('t(s)', 'i_term'),
                                     ('t(s)', 'd_term'))

  crappy.link(pid, graph_pid2, modifier=crappy.modifier.Mean(10))

  crappy.start()
