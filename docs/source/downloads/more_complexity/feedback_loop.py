# coding: utf-8

# [feedback-loop-start]
from os import environ

import crappy


def main() -> None:
  target = crappy.blocks.Generator(
      path=(
        {'type': 'Constant', 'value': 500, 'condition': 'delay=2'},
        {'type': 'Constant', 'value': 1500, 'condition': 'delay=3'},
        {'type': 'Constant', 'value': 500, 'condition': 'delay=2'},
      ),
      cmd_label='target_speed',
      spam=True,
      freq=30,
      end_delay=0.2)

  motor = crappy.blocks.Machine(
      ({'type': 'FakeDCMotor',
        'cmd_label': 'voltage',
        'mode': 'speed',
        'speed_label': 'actual_speed',
        'kv': 1000,
        'inertia': 4,
        'rv': 0.2,
        'fv': 1e-5},),
      freq=30)

  controller = crappy.blocks.PID(
      kp=0.038,
      ki=0.076,
      kd=0.0019,
      out_min=-10,
      out_max=10,
      i_limit=(-5, 5),
      setpoint_label='target_speed',
      input_label='actual_speed',
      labels=('t(s)', 'voltage'),
      freq=30)

  graph = crappy.blocks.Grapher(
      ('t(s)', 'target_speed'),
      ('t(s)', 'actual_speed'),
      backend=environ.get('MPLBACKEND', 'TkAgg'))

  crappy.link(target, controller)
  crappy.link(motor, controller)
  crappy.link(controller, motor)

  crappy.link(target, graph)
  crappy.link(motor, graph)

  crappy.start()


if __name__ == '__main__':
  main()
# [feedback-loop-end]
