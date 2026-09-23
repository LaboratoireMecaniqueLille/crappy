# coding: utf-8

# [custom-inout-start]
from time import time

import crappy


# [custom-inout-class-start]
class SimulatedInOut(crappy.inout.InOut):

  def __init__(self, sensor_offset: float = 0) -> None:
    super().__init__()
    self._sensor_offset = sensor_offset
    self._command = 0.0
    self._is_open = False

  def open(self) -> None:
    self._is_open = True

  def get_data(self) -> tuple[float, float]:
    measured_value = self._sensor_offset + self._command
    return time(), measured_value

  def set_cmd(self, target_value: float) -> None:
    self._command = target_value

  def close(self) -> None:
    self._command = 0.0
    self._is_open = False
# [custom-inout-class-end]


def main() -> None:
  command = crappy.blocks.Generator(
      path=({'type': 'Cyclic',
             'value1': 0.5,
             'condition1': 'delay=1',
             'value2': -0.5,
             'condition2': 'delay=1',
             'cycles': 2},),
      cmd_label='target_value',
      freq=10,
      end_delay=0.2)

  # [custom-inout-use-start]
  instrument = crappy.blocks.IOBlock(
      name='SimulatedInOut',
      labels=('t(s)', 'measured_value'),
      cmd_labels=('target_value',),
      initial_cmd=(0,),
      exit_cmd=(0,),
      make_zero_delay=0.2,
      sensor_offset=1.5,
      freq=10)
  # [custom-inout-use-end]

  reader = crappy.blocks.LinkReader(name='Instrument reading', freq=10)

  crappy.link(command, instrument)
  crappy.link(instrument, reader)

  crappy.start()


if __name__ == '__main__':
  main()
# [custom-inout-end]
