# coding: utf-8

# [actuator-control-start]
import crappy


def main() -> None:
  command = crappy.blocks.Generator(
      path=({'type': 'Cyclic',
             'value1': 6,
             'condition1': 'delay=1',
             'value2': -6,
             'condition2': 'delay=1',
             'cycles': 2},),
      cmd_label='voltage(V)',
      freq=20,
      end_delay=0.2)

  motor = crappy.blocks.Machine(
      ({'type': 'FakeDCMotor',
        'cmd_label': 'voltage(V)',
        'mode': 'speed',
        'speed_label': 'speed(RPM)',
        'position_label': 'position(turns)',
        'kv': 500},),
      freq=20)

  reader = crappy.blocks.LinkReader(name='Motor state', freq=20)

  crappy.link(command, motor)
  crappy.link(motor, reader)

  crappy.start()


if __name__ == '__main__':
  main()
# [actuator-control-end]
