#coding: utf-8

import crappy

g = crappy.blocks.Generator([dict(type='sine',freq=1,amplitude=1,condition=None)])
s = crappy.blocks.Server()
crappy.link(g,s)

crappy.start()
