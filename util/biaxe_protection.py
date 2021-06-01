# coding: utf-8

import crappy

gx = crappy.blocks.Generator([{'type': 'protection', 'condition2': 'F3>5',
                               'value1': 10, 'condition1': 'F3<-5',
                               'value2': -10}], cmd_label='vx')

gy = crappy.blocks.Generator([{'type': 'protection', 'condition2': 'F1>5',
                               'value1': 10, 'condition1': 'F1<-5',
                               'value2': -10}], cmd_label='vy')

m1 = {'port': '/dev/ttyS4', 'cmd': 'vy'}
m2 = {'port': '/dev/ttyS5', 'cmd': 'vy'}
m3 = {'port': '/dev/ttyS6', 'cmd': 'vx'}
m4 = {'port': '/dev/ttyS7', 'cmd': 'vx'}
common = {'type': 'biaxe'}

m = crappy.blocks.Machine([m1, m2, m3, m4], common)
s = crappy.blocks.IOBlock('Comedi', channels=[1, 3], gain=[3749, 3749],
      labels=['t(s)', 'F1', 'F3'])

g = crappy.blocks.Grapher(('t(s)', 'F1'), ('t(s)', 'F3'))

crappy.link(s, g)
crappy.link(s, gx)
crappy.link(s, gy)
crappy.link(gx, m)
crappy.link(gy, m)

crappy.start()
