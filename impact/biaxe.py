# coding: utf-8

"""
This program is used to control a biaxial tensile machine.

It demonstrates an equibiaxial cyclic test.
"""

import crappy

path = {'type': 'cyclic', 'value1': 5, 'condition1': 'delay=3',
                        'value2': -5, 'condition2': 'delay=3', 'cycles': 0}
path2 = dict(path)
path2['condition1'] = 'delay=5'
path2['condition2'] = 'delay=5'
g1 = crappy.blocks.Generator(path=[path], cmd_label="vx")
g2 = crappy.blocks.Generator(path=[path2], cmd_label="vy")

mot = {'type': 'biaxe',
       'mode': 'speed'
       }

motA = {'port': '/dev/ttyS4',
        'cmd': 'vy',
        }

motB = {'port': '/dev/ttyS5',
        'cmd': 'vx',
        }

motC = {'port': '/dev/ttyS6',
        'cmd': 'vy',
        }

motD = {'port': '/dev/ttyS7',
        'cmd': 'vx',
        }

b = crappy.blocks.Machine([motA, motB, motC, motD], common=mot)
crappy.link(g1, b)
crappy.link(g2, b)
crappy.start()
