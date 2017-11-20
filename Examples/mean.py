#coding: utf-8

import crappy

g1 = crappy.blocks.Generator(
    [dict(type='sine',freq=2,amplitude=2,condition=None)],
    freq=200,
    cmd_label = 'cmd1'
    )

g2 = crappy.blocks.Generator(
    [dict(type='sine',freq=.2,amplitude=2,condition=None)],
    freq=200,
    cmd_label = 'cmd2'
    )

m = crappy.blocks.Mean(.5)#,out_labels=['cmd1','cmd2'])

crappy.link(g1,m)
crappy.link(g2,m)

g = crappy.blocks.Grapher(('t(s)','cmd1'),('t(s)','cmd2'))
crappy.link(m,g)
crappy.start()
