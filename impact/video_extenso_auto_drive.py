# coding: utf-8

"""
Code demonstrating the use of a linear actuator to follow the videoextensometry
markers during a test with large strains.
"""

import crappy

ve = crappy.blocks.Video_extenso(camera='Ximea_cv', show_image=True)

ad = crappy.blocks.AutoDrive(
    actuator={'name': 'CM_drive', 'port': '/dev/ttyUSB0'}, direction='X-')

graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))

crappy.link(ve, graph_extenso)

crappy.link(ve, ad)

crappy.start()
