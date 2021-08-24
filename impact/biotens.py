# coding: utf-8

"""
This program is often used as the starting point when performing tests on the
"Biotens" machine.

It creates a new folder for each experiment and performs tensile tests using
videoextensometry.
"""

import time

import crappy

save_path = "biotens_data/"
timestamp = time.ctime()[:-5].replace(" ", "_")
save_path += timestamp + "/"

# Creating F sensor
effort = crappy.blocks.IOBlock("Comedi", channels=[0], gain=[-48.8],
                               labels=['t(s)', 'F(N)'])

# grapher
graph_effort = crappy.blocks.Grapher(('t(s)', 'F(N)'))
crappy.link(effort, graph_effort)

# and recorder
rec_effort = crappy.blocks.Recorder(save_path + "effort.csv")
crappy.link(effort, rec_effort)

# Quick hack to reset the position of the actuator
b = crappy.actuator.Biotens()
b.open()
b.reset_position()
b.set_position(5, 50)

# Creating Machine block...
biotens = crappy.blocks.Machine([{'type': 'biotens',
                                  'port': '/dev/ttyUSB0',
                                  'pos_label': 'position1',
                                  'cmd': 'cmd'}])
# ..graph...
# graph_pos = crappy.blocks.Grapher(('t(s)', 'position1'))
# crappy.link(biotens, graph_pos)
# ...and recorder
rec_pos = crappy.blocks.Recorder(save_path + 'position.csv')
crappy.link(biotens, rec_pos)

# To pilot the biotens
generator = crappy.blocks.Generator([{'type': 'constant',
                                      'condition': 'F(N)>90',
                                      'value': 5}], freq=100)
crappy.link(effort, generator)
crappy.link(generator, biotens)

# VideoExtenso
extenso = crappy.blocks.Video_extenso(camera="Ximea_cv", white_spots=False)

# Recorder
rec_extenso = crappy.blocks.Recorder(save_path + 'extenso.csv',
                                     labels=['t(s)', 'Exx(%)', 'Eyy(%)'])
crappy.link(extenso, rec_extenso)
# And grapher
graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))
crappy.link(extenso, graph_extenso)

# And here we go !
crappy.start()
