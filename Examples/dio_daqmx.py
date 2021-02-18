#coding: utf-8

import crappy


def intify(data):
  for i,d in enumerate(data):
    if isinstance(d,bool):
      data[i] = int(d)
  return data

if __name__ == "__main__":
  gen = crappy.blocks.Generator([
  dict(type='cyclic',value1=0,value2=1,condition1="delay=1",condition2="delay=1")
  ],repeat=True)
  io = crappy.blocks.IOBlock("Nidaqmx",device="Dev2",
      channels=[dict(name='ai0'),dict(name='di0'),dict(name='ao0'),dict(name='do1')],
      samplerate = 100,
      labels = ['t(s)','ai0','di0'],
      cmd_labels = ['cmd','cmd'],
      )
  crappy.link(gen,io)
  graph = crappy.blocks.Grapher(('t(s)','di0'),('t(s)','ai0'))
  crappy.link(io,graph,modifier=intify)
  crappy.start()
