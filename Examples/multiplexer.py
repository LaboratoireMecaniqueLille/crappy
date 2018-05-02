#coding: utf-8

import crappy

g1 = crappy.blocks.Generator([
  dict(type='sine',freq=1,amplitude=1,condition=None)
  ],freq=100,cmd_label='cmd1')

g2 = crappy.blocks.Generator([
  dict(type='cyclic_ramp',speed2=-1,speed1=1,
    condition1='cmd2>1',condition2='cmd2<-1',cycles=1e30)
  ],freq=50,cmd_label='cmd2')

mul = crappy.blocks.Multiplex()

crappy.link(g1,mul)
crappy.link(g2,mul)

graph = crappy.blocks.Grapher(('t(s)','cmd1'),('t(s)','cmd2'))

crappy.link(mul,graph)

save = crappy.blocks.Saver("example_multi.csv",labels=["t(s)", "cmd1", "cmd2"])

crappy.link(mul,save)

crappy.start()
