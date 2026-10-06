# coding: utf-8

# [quickstart-start]
import crappy


def main() -> None:
  command = crappy.blocks.Generator(
      path=({'type': 'Constant',
             'value': 0.5,
             'condition': 'delay=3'},),
      cmd_label='input_speed',
      freq=10,
      end_delay=0.2)

  machine = crappy.blocks.FakeMachine(cmd_label='input_speed', freq=5)
  reader = crappy.blocks.LinkReader(name='Measurements', freq=10)

  crappy.link(command, machine)
  crappy.link(machine, reader)

  crappy.start()


if __name__ == '__main__':
  main()
# [quickstart-end]
