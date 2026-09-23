# coding: utf-8

# [command-generation-start]
import crappy


def main() -> None:
  command = crappy.blocks.Generator(
      path=({'type': 'Ramp',
             'speed': 1,
             'init_value': 0,
             'condition': 'delay=2'},
            {'type': 'Constant',
             'value': 2,
             'condition': 'delay=2'}),
      cmd_label='target',
      spam=True,
      freq=5,
      end_delay=0.2)

  reader = crappy.blocks.LinkReader(name='Generated command', freq=10)

  crappy.link(command, reader)

  crappy.start()


if __name__ == '__main__':
  main()
# [command-generation-end]
