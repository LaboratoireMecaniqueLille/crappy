# coding: utf-8

"""
Basic example demonstrating the use of the DISCorrel block.

Required hardware:
  - Any camera
"""

import crappy

if __name__ == "__main__":
  dis = crappy.blocks.DISCorrel('Webcam',
      fields=['x', 'y'],
      labels=['t(s)', 'x(pix)', 'y(pix)'])

  graph = crappy.blocks.Grapher(('x(pix)', 'y(pix)'))
  crappy.link(dis, graph)
  crappy.start()
