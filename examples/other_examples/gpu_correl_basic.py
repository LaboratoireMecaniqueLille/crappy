# coding: utf-8

"""
A basic example showing how to use the GPUCorrel block.

Required hardware:
  - Any camera
  - A CUDA compatible GPU
"""

import crappy

if __name__ == "__main__":

  # Creating the GPUCorrel block
  correl = crappy.blocks.GPUCorrel(camera="Webcam",
                                   fields=['x', 'y', 'r'],  # Rigid body
                                   img_dtype='uint8',
                                   img_shape=(480, 640))

  # Creating the Grapher Block for displaying the field values
  graph = crappy.blocks.Grapher(('t(s)', 'x'), ('t(s)', 'y'), ('t(s)', 'r'),
                                length=50)
  crappy.link(correl, graph)

  # Starting the test
  crappy.start()
