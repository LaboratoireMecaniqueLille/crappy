# coding: utf-8

# [custom-generator-path-start]
from time import time

import crappy


# [custom-generator-path-class-start]
class TimedPulse(crappy.Path):

  def __init__(self,
               low: float,
               high: float,
               period: float,
               high_time: float,
               condition) -> None:
    super().__init__()

    if period <= 0:
      raise ValueError('period must be positive')
    if not 0 <= high_time <= period:
      raise ValueError('high_time must lie between zero and period')

    self._low = low
    self._high = high
    self._period = period
    self._high_time = high_time
    self._condition = self.parse_condition(condition)

  def get_cmd(self, data: dict[str, list]) -> float:
    if self._condition(data):
      raise StopIteration

    elapsed = time() - self.t0
    if elapsed % self._period < self._high_time:
      return self._high
    return self._low
# [custom-generator-path-class-end]


def main() -> None:
  # [custom-generator-path-use-start]
  command = crappy.blocks.Generator(
      path=({'type': 'TimedPulse',
             'low': 0,
             'high': 5,
             'period': 1,
             'high_time': 0.2,
             'condition': 'delay=4'},),
      cmd_label='pulse(V)',
      spam=True,
      freq=10,
      end_delay=0.2)
  # [custom-generator-path-use-end]

  reader = crappy.blocks.LinkReader(name='Pulse command', freq=10)
  crappy.link(command, reader)

  crappy.start()


if __name__ == '__main__':
  main()
# [custom-generator-path-end]
