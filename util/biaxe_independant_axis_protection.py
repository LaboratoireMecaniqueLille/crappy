# coding: utf-8

import crappy

g1 = crappy.blocks.Generator([{'type': 'protection', 'condition2': 'F1>5',
                               'value1': 10, 'condition1': 'F1<-5',
                               'value2': -10}], cmd_label='v1')

g2 = crappy.blocks.Generator([{'type': 'protection', 'condition2': 'F2>5',
                               'value1': 10, 'condition1': 'F2<-5',
                               'value2': -10}], cmd_label='v2')

g3 = crappy.blocks.Generator([{'type': 'protection', 'condition2': 'F3>5',
                               'value1': 10, 'condition1': 'F3<-5',
                               'value2': -10}], cmd_label='v3')

g4 = crappy.blocks.Generator([{'type': 'protection', 'condition2': 'F4>5',
                               'value1': 10, 'condition1': 'F4<-5',
                               'value2': -10}], cmd_label='v4')

m1 = {'port': '/dev/ttyS4', 'cmd': 'v1'}
m2 = {'port': '/dev/ttyS5', 'cmd': 'v2'}
m3 = {'port': '/dev/ttyS6', 'cmd': 'v3'}
m4 = {'port': '/dev/ttyS7', 'cmd': 'v4'}
common = {'type': 'biaxe'}

m = crappy.blocks.Machine([m1, m2, m3, m4], common)
s = crappy.blocks.IOBlock('Comedi', channels=[1, 2, 3, 4], gain=3749,
                          labels=['t(s)', 'F1', 'F2', 'F3', 'F4'])

g = crappy.blocks.Grapher(('t(s)', 'F1'), ('t(s)', 'F2'),
                          ('t(s)', 'F3'), ('t(s)', 'F4'))

crappy.link(s, g)
crappy.link(s, g1)
crappy.link(s, g2)
crappy.link(s, g3)
crappy.link(s, g4)
crappy.link(g1, m)
crappy.link(g2, m)
crappy.link(g3, m)
crappy.link(g4, m)

crappy.start()
