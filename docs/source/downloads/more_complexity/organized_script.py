# coding: utf-8

# [organized-script-start]
from pathlib import Path
from tempfile import mkdtemp

import crappy


def main() -> None:
  output_dir = Path(mkdtemp(prefix='crappy-organized-'))
  print(f'The recordings will be saved in:\n{output_dir}')

  speed_steps = (0.5, 1.0, 0.0)
  paths = tuple(
      {'type': 'Constant',
       'value': speed,
       'condition': 'delay=1'}
      for speed in speed_steps)

  command = crappy.blocks.Generator(
      path=paths,
      cmd_label='input_speed',
      freq=10,
      end_delay=0.2)

  machine = crappy.blocks.FakeMachine(
      cmd_label='input_speed',
      freq=10)

  measurements_to_record = {
    'force.csv': 'F(N)',
    'position.csv': 'x(mm)',
    'strain.csv': 'Exx(%)',
  }

  recorders = list()
  for file_name, label in measurements_to_record.items():
    recorder = crappy.blocks.Recorder(
        file_name=output_dir / file_name,
        labels=('t(s)', label),
        delay=0.2,
        freq=10)
    recorders.append(recorder)
    crappy.link(machine, recorder)

  crappy.link(command, machine)

  crappy.start()
  print(f'Recorded files are available in:\n{output_dir}')


if __name__ == '__main__':
  main()
# [organized-script-end]
