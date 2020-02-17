#coding: utf-8

import crappy

graph = crappy.blocks.Grapher(('t(s)','x'),('t(s)','y'),('t(s)','r'),length=50)

correl = crappy.blocks.GPUCorrel(camera="Webcam",fields=['x','y','r','exx','eyy']) # Rigid body

crappy.link(correl,graph)

crappy.start()
