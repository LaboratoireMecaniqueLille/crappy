# coding: utf-8

# [streaming-acquisition-start]
from pathlib import Path
from tempfile import mkdtemp

import crappy


def main() -> None:
  output_file = Path(mkdtemp(prefix='crappy-stream-')) / 'memory.h5'
  print(f'The stream will be saved to:\n{output_file}')

  stream = crappy.blocks.IOBlock(
      'FakeInOut',
      labels=('t(s)', 'stream'),
      streamer=True,
      freq=10)

  recorder = crappy.blocks.HDFRecorder(
      filename=output_file,
      label='stream',
      atom='float64',
      expected_rows=500,
      flush_period=5)

  reader = crappy.blocks.LinkReader(
      name='Mean memory usage',
      freq=5)

  stop = crappy.blocks.StopBlock('t(s) > 3')

  crappy.link(stream, recorder)
  crappy.link(
      stream,
      reader,
      modifier=crappy.modifier.Demux(
          labels='memory(%)',
          stream_label='stream',
          mean=True))

  crappy.start()
  print(f'Recorded stream is available at:\n{output_file}')


if __name__ == '__main__':
  main()
# [streaming-acquisition-end]
