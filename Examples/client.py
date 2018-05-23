#coding: utf-8

import crappy

s = crappy.blocks.Client('localhost')
g = crappy.blocks.Grapher(('t(s)','cmd'))

crappy.link(s,g)

crappy.start()
