# coding: utf-8

# [generator-conditions-start]
import crappy


def main() -> None:
  command = crappy.blocks.Generator(
      path=(
        {'type': 'Constant',
         'value': 1,
         'condition': 'x(mm) > 1'},
        {'type': 'Constant',
         'value': 0,
         'condition': 'delay=1'},
      ),
      cmd_label='input_speed',
      safe_start=True,
      spam=True,
      freq=20,
      end_delay=0.2)

  machine = crappy.blocks.FakeMachine(
      cmd_label='input_speed',
      freq=20)

  reader = crappy.blocks.LinkReader(
      name='Machine measurements',
      freq=10)

  crappy.link(command, machine)
  crappy.link(machine, command)
  crappy.link(machine, reader)

  crappy.start()


if __name__ == '__main__':
  main()
# [generator-conditions-end]
