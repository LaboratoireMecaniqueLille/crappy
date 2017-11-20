#coding: utf-8
from __future__ import absolute_import
import crappy

if __name__ == "__main__":
  sg1 = crappy.blocks.Generator([{'type':'sine','freq':.5,'amplitude':2,
    'offset':1,'condition':'delay=1000'}],cmd_label='cmd1',freq=500)

  sg2 = crappy.blocks.Generator([{'type':'sine','freq':.8,'amplitude':2,
    'offset':3,'condition':'delay=1000'}],cmd_label='cmd2',freq=500)

  io = crappy.blocks.IOBlock("Labjack_t7",labels=['t(s)','c0','c1'],
      cmd_labels = ['cmd1','cmd2'],
      channels = [
      {'name':'AIN0'},
      {'name':'AIN1'},
      {'name':'DAC0'},
      {'name':'DAC1'},
      ],
      verbose=True)

  crappy.link(sg1,io)
  crappy.link(sg2,io)

  g = crappy.blocks.Grapher(('t(s)','c0'),('t(s)','c1'))

  crappy.link(io,g)

  crappy.start()
