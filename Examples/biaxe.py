#coding: utf-8

import crappy

path = {'type':'cyclic','value1':20,'condition1':'delay=10',
                        'value2':-20,'condition2':'delay=10','cycles':0}
path2 = dict(path)
path2['condition1'] = 'delay=15'
path2['condition2'] = 'delay=15'
g1 = crappy.blocks.Generator(path=[path],cmd_label="vx")
g2 = crappy.blocks.Generator(path=[path2],cmd_label="vy")

mot = {'type':'oriental',
       'mode':'speed'
       }

motA = {'port':'/dev/ttyUSB2',
        'cmd':'vy',
        'pos_label':'posA'
        }

motB = {'port':'/dev/ttyUSB3',
        'cmd':'vx',
        'pos_label':'posB'
        }

motC = {'port':'/dev/ttyUSB0',
        'cmd':'vy',
        'pos_label':'posC'
        }

motD = {'port':'/dev/ttyUSB1',
        'cmd':'vx',
        'pos_label':'posD'
        }

b = crappy.blocks.Machine([motA,motB,motC,motD],common=mot)
crappy.link(g1,b)
crappy.link(g2,b)
crappy.start()
