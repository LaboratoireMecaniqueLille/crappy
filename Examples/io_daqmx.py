#coding: utf-8

import crappy

if __name__ == "__main__":
  #"""
  gen = crappy.blocks.Generator([
  dict(type='sine',freq=1,amplitude=2,offset=1,condition=None)
  ])
  io = crappy.blocks.IOBlock("Nidaqmx",device="Dev2",
      channels=[dict(name='ai0'),dict(name='ao0')],
      samplerate = 100,
      labels = ['t(s)','ai0'],
      cmd_labels = ['cmd'],
      )
  crappy.link(gen,io)
  graph = crappy.blocks.Grapher(('t(s)','ai0'))
  crappy.link(io,graph)
  crappy.start()
  """

  io = crappy.blocks.IOBlock("Nidaqmx",device="Dev2",
      channels=[dict(name='ai0')],
      samplerate = 100,
      labels = ['t(s)','ai0'],
      )

  graph = crappy.blocks.Grapher(('t(s)','ai0'))
  crappy.link(io,graph)
  crappy.start()
  #"""