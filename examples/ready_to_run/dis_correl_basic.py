# coding: utf-8

"""
Basic example demonstrating the use of the DISCorrel block.

Required hardware:
  - Any camera
"""

import crappy

if __name__ == "__main__":

  # The Block acquiring the images and performing the image correlation
  dis = crappy.blocks.DISCorrel('Webcam',
                                fields=['x', 'y'],
                                labels=['t(s)', 'meta', 'x(pix)', 'y(pix)'],
                                display_images=True)

  # The Block displaying the calculated values
  graph = crappy.blocks.Grapher(('x(pix)', 'y(pix)'))
  crappy.link(dis, graph)

  # Starting the test
  crappy.start()
