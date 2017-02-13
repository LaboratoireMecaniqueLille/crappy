#coding: utf-8

import crappy

ve = crappy.blocks.VideoExtenso(camera='Ximea', compacter=30)

ad = crappy.blocks.AutoDrive(technical='CmDrive',dev_args={'port': '/dev/ttyUSB0'},direction='Y-')

graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'),
				      length=0)

crappy.link(ve,graph_extenso)

crappy.link(ve,ad)

crappy.start()