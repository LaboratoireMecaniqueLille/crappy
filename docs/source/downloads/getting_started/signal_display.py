# coding: utf-8

# [signal-display-start]
import crappy


def main() -> None:
  command = crappy.blocks.Generator(
      path=({'type': 'Constant',
             'value': 0.5,
             'condition': 'delay=5'},),
      cmd_label='input_speed',
      freq=20,
      end_delay=0.2)

  machine = crappy.blocks.FakeMachine(cmd_label='input_speed', freq=20)
  graph = crappy.blocks.Grapher(('t(s)', 'F(N)'))

  crappy.link(command, machine)
  crappy.link(machine, graph)

  crappy.start()


if __name__ == '__main__':
  main()
# [signal-display-end]
