# coding: utf-8

import crappy

if __name__ == '__main__':

  gen = crappy.blocks.Generator([
    {'type': 'Constant', 'value': 1000, 'condition': 'delay=3'},
    {'type': 'Ramp', 'speed': 100, 'condition': 'delay=5', 'init_value': 0},
    {'type': 'Constant', 'value': 1800, 'condition': 'delay=3'},
    {'type': 'Constant', 'value': 500, 'condition': 'delay=3'},
    {'type': 'Sine', 'amplitude': 2000, 'offset': 1000, 'freq': .3,
     'condition': 'delay=15'}],
      spam=True,
      cmd_label='target_speed')

  mot = crappy.blocks.Machine([{'type': 'FakeDCMotor',
                                'cmd_label': 'voltage',
                                'mode': 'speed',
                                'speed_label': 'actual_speed',
                                'kv': 1000,
                                'inertia': 4,
                                'rv': .2,
                                'fv': 1e-5}])

  graph = crappy.blocks.Grapher(('t(s)', 'actual_speed'),
                                ('t(s)', 'target_speed'))

  pid = crappy.blocks.PID(kp=0.038,
                          ki=0.076,
                          kd=0.0019,
                          out_max=10,
                          out_min=-10,
                          i_limit=(-5, 5),
                          setpoint_label='target_speed',
                          labels=('t(s)', 'voltage'),
                          input_label='actual_speed')

  crappy.link(gen, pid)
  crappy.link(mot, pid)

  crappy.link(pid, mot)

  crappy.link(gen, graph)
  crappy.link(mot, graph)

  crappy.start()
