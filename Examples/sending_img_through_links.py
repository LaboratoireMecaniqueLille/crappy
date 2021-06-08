# coding: utf-8"

"""
Example demonstrating how to use the ``input_label`` to send images between
blocks

This mechanism is useful to perform fake tests by using using generated images
instead of cameras to read images.

Required hardware:
  - Any camera
"""

import crappy

if __name__ == "__main__":
  cam1 = crappy.blocks.Camera('Webcam')

  dis = crappy.blocks.DISCorrel('', input_label='frame')
  crappy.link(cam1, dis)

  graph = crappy.blocks.Grapher(('t(s)', 'x(pix)'))
  crappy.link(dis, graph)

  crappy.start()
