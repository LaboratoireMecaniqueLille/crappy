#coding: utf-8

import crappy

x = y = 2048

camera = crappy.blocks.StreamerCamera(camera="ximea",width=x,height=y)

graph = crappy.blocks.Grapher(('t','x'),('t','y'),('t','r'),length=50)

correl = crappy.blocks.Correl((y,x),fields=['x','y','r']) # Rigid body

crappy.link(camera,correl)
crappy.link(correl,graph)

crappy.start()
