# coding: utf-8

"""
Demonstration of a Videoextensometry controlled test.

This program is intended as a demonstration and is fully virtual.

No hardware required
"""

import crappy
import cv2

if __name__ == "__main__":
  img = cv2.imread('data/ve_markers.tif', 0)

  speed = 5 / 60  # mm/sec

  # Load until the strain is reached, then unload until force is 0
  generator = crappy.blocks.Generator(path=sum([[
    {'type': 'constant', 'value': speed,
     'condition': 'Exx(%)>{}'.format(5 * i)},
    {'type': 'constant', 'value': -speed, 'condition': 'F(N)<0'}]
    for i in range(1, 5)], []), spam=False)

  # Our fake machine
  machine = crappy.blocks.Fake_machine(maxstrain=17, k=5000, l0=20,
      plastic_law=lambda exx: 0, sigma={'F(N)': 0.5})

  crappy.link(generator, machine)
  crappy.link(machine, generator)

  # The block performing the videoextensometry
  ve = crappy.blocks.Video_extenso('', input_label='frame', show_image=True,
      blur=False)
  # This modifier will generate an image with the values of strain
  # coming from the Fake_machine block
  crappy.link(machine, ve, modifier=crappy.modifier.Apply_strain_img(img))

  graph_def2 = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))
  crappy.link(ve, graph_def2)

  crappy.start()
