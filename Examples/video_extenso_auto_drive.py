#coding: utf-8

import crappy

ve = crappy.blocks.Video_extenso(camera='XimeaCV',show_image=True)

ad = crappy.blocks.AutoDrive(
    actuator={'name':'CM_drive','port': '/dev/ttyUSB0'},direction='X-')

graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'),
				      length=0)

crappy.link(ve,graph_extenso)

crappy.link(ve,ad)

crappy.start()
