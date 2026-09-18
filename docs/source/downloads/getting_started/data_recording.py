# coding: utf-8

# [data-recording-start]
from pathlib import Path
from tempfile import mkdtemp

import crappy


def main() -> None:
  output_file = Path(mkdtemp(prefix='crappy-recorder-')) / 'measurements.csv'
  print(f'The recording will be saved to:\n{output_file}')

  command = crappy.blocks.Generator(
      path=({'type': 'Constant',
             'value': 0.5,
             'condition': 'delay=3'},),
      cmd_label='input_speed',
      freq=10,
      end_delay=0.2)

  machine = crappy.blocks.FakeMachine(cmd_label='input_speed', freq=10)

  recorder = crappy.blocks.Recorder(
      file_name=output_file,
      labels=('t(s)', 'F(N)', 'x(mm)'),
      delay=0.2,
      freq=10)

  crappy.link(command, machine)
  crappy.link(machine, recorder)

  crappy.start()
  print(f'Recorded data is available at:\n{output_file}')


if __name__ == '__main__':
  main()
# [data-recording-end]
