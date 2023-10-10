# coding: utf-8

import crappy

if __name__ == '__main__':

  io = crappy.blocks.IOBlock('FakeInOut',
                             labels=('t(s)', 'stream'),
                             streamer=True,
                             freq=30)

  rec = crappy.blocks.HDFRecorder(filename='data.hdf5',
                                  label='stream',
                                  atom='float64')

  graph = crappy.blocks.Grapher(('t(s)', 'memory'))

  stop = crappy.blocks.StopButton()

  crappy.link(io, rec)
  crappy.link(io, graph,
              modifier=crappy.modifier.Demux(labels='memory',
                                             stream_label='stream',
                                             mean=True))

  crappy.start()
