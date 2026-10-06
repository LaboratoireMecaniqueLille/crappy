# coding: utf-8

# [custom-streaming-inout-start]
from time import time

import numpy as np

import crappy


# [custom-streaming-inout-class-start]
class SimulatedStream(crappy.inout.InOut):

  def __init__(self, sample_rate: float = 50, chunk_size: int = 5) -> None:
    super().__init__()
    if sample_rate <= 0 or chunk_size <= 0:
      raise ValueError('sample_rate and chunk_size must be positive')
    self._sample_rate = sample_rate
    self._chunk_size = chunk_size
    self._start_time = 0.0
    self._sample_index = 0
    self._streaming = False

  def open(self) -> None:
    self._sample_index = 0

  def start_stream(self) -> None:
    self._start_time = time()
    self._sample_index = 0
    self._streaming = True

  def get_stream(self) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(
        self._sample_index,
        self._sample_index + self._chunk_size)
    timestamps = self._start_time + indices / self._sample_rate
    signal = np.sin(2 * np.pi * indices / self._sample_rate)
    self._sample_index += self._chunk_size
    return timestamps, signal[:, np.newaxis]

  def stop_stream(self) -> None:
    self._streaming = False

  def close(self) -> None:
    self._streaming = False
# [custom-streaming-inout-class-end]


def main() -> None:
  # [custom-streaming-inout-use-start]
  stream = crappy.blocks.IOBlock(
      name='SimulatedStream',
      labels=('t(s)', 'stream'),
      streamer=True,
      sample_rate=50,
      chunk_size=5,
      freq=10)
  # [custom-streaming-inout-use-end]

  reader = crappy.blocks.LinkReader(name='Mean stream value', freq=10)
  stop = crappy.blocks.StopBlock('t(s) > 3')

  crappy.link(
      stream,
      reader,
      modifier=crappy.modifier.Demux(labels=('signal',), mean=True))
  crappy.link(
      stream,
      stop,
      modifier=crappy.modifier.Demux(labels=('signal',), mean=True))

  crappy.start()


if __name__ == '__main__':
  main()
# [custom-streaming-inout-end]
