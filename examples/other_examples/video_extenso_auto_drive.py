# coding: utf-8

"""
Code demonstrating the use of a linear actuator to follow the videoextensometry
markers during a test with large strains.
"""

import crappy

if __name__ == '__main__':

  # The Block acquiring the images and performing video-extensometry
  ve = crappy.blocks.VideoExtenso(camera='XiAPI', display_images=True)

  # The Block driving the Actuator for following the spots
  auto_drive = crappy.blocks.AutoDriveVideoExtenso(
      actuator={'name': 'SchneiderMDrive23', 'port': '/dev/ttyUSB0'},
      direction='X-')
  crappy.link(ve, auto_drive)

  # The Block displaying the strain in real-time
  graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))
  crappy.link(ve, graph_extenso)

  # Starting the test
  crappy.start()
