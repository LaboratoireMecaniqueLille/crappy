# coding: utf-8

"""
Demonstration of a DIC controlled test.

This program is intended as a demonstration and is fully virtual.

Required hardware:
  - A CUDA compatible GPU (and pycuda installed)
Requires the cv2 module to be installed.
"""

import crappy

if __name__ == "__main__":
  img = crappy.resources.speckle

  speed = 5 / 60  # mm/sec

  # Load until the strain is reached, then unload until force is 0
  generator = crappy.blocks.Generator(path=sum([[
    {'type': 'constant', 'value': speed,
     'condition': 'Exx(%)>{}'.format(5 * i)},
    {'type': 'constant', 'value': -speed, 'condition': 'F(N)<0'}]
    for i in range(1, 5)], []), spam=False)

  # Our fake machine
  machine = crappy.blocks.Fake_machine(max_strain=17, k=5000, l0=20,
                                       plastic_law=lambda exx: 0,
                                       sigma={'F(N)': 0.5})

  crappy.link(generator, machine)
  crappy.link(machine, generator)

  # The block performing the DIC
  dis = crappy.blocks.GPUCorrel('', input_label='frame', verbose=True,
                                labels=['x', 'y', 'measured_Exx(%)',
                                        'measured_Eyy(%)'],
                                fields=['x', 'y', 'exx', 'eyy'], levels=3)
  # This modifier will generate an image with the values of strain
  # coming from the Fake_machine block
  crappy.link(machine, dis, modifier=crappy.modifier.Apply_strain_img(img))

  graph_def2 = crappy.blocks.Grapher(('t(s)', 'measured_Exx(%)'),
                                     ('t(s)', 'measured_Eyy(%)'))
  crappy.link(dis, graph_def2)

  crappy.start()
