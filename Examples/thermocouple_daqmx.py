#coding: utf-8

import crappy

STREAMER = True
channels = range(1)
chan_names = ['ai%d'%i for i in channels]
labels = ['t(s)']+['T%d'%i for i in channels]

if __name__ == "__main__":
  
  io = crappy.blocks.IOBlock("Nidaqmx",device="cDAQ1Mod1",
      channels=[dict(name=c_name,units='C',
      type='thrmcpl',
      thermocouple_type='K') for c_name in chan_names],
      samplerate = 14./len(channels),
      labels = ['t(s)','stream'] if STREAMER else labels,
      streamer = STREAMER
      )

  graph = crappy.blocks.Grapher(*[('t(s)',l) for l in labels[1:]])
  if STREAMER:
    crappy.link(io,graph,condition=crappy.condition.Demux(labels[1:],mean=False,transpose=True))
  else:
    crappy.link(io,graph)
  crappy.start()