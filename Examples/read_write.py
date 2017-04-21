#coding: utf-8
from __future__ import absolute_import
import crappy


for i,c in enumerate(crappy.inout.inout_list):
  print i,c
name = crappy.inout.inout_list.keys()[int(raw_input(
                    "What board do you want to use ?> "))]

sg = crappy.blocks.Generator([{'type':'sine','freq':.5,'amplitude':1,
  'offset':.5,'condition':'delay=1000'}])

io = crappy.blocks.IOBlock(name,labels=['t(s)','chan0'],
    out_channels=0,verbose=True)
crappy.link(sg,io)

g = crappy.blocks.Grapher(('t(s)','chan0'))

crappy.link(io,g)

crappy.start()
