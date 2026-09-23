# coding: utf-8

# [custom-block-start]
from statistics import fmean
from time import time

import crappy


# [custom-block-class-start]
class BatchAverage(crappy.blocks.Block):

  def __init__(self,
               input_label: str,
               output_label: str,
               freq: float = 2) -> None:
    super().__init__()
    self._input_label = input_label
    self._output_label = output_label
    self.freq = freq

  def loop(self) -> None:
    received = self.recv_all_data()
    values = received.get(self._input_label, [])

    if values:
      self.send({'t(s)': time() - self.t0,
                 self._output_label: fmean(values),
                 'sample_count': len(values)})
# [custom-block-class-end]


def main() -> None:
  force = crappy.blocks.Generator(
      path=({'type': 'Sine',
             'amplitude': 10,
             'freq': 1,
             'condition': 'delay=4'},),
      cmd_label='force(N)',
      freq=20,
      end_delay=0.2)

  # [custom-block-use-start]
  average = BatchAverage(
      input_label='force(N)',
      output_label='mean_force(N)',
      freq=2)
  # [custom-block-use-end]

  reader = crappy.blocks.LinkReader(name='Batch result', freq=5)

  crappy.link(force, average)
  crappy.link(average, reader)

  crappy.start()


if __name__ == '__main__':
  main()
# [custom-block-end]
