# coding: utf-8

"""
A basic example showing how to use the GPUCorrel block.

Required hardware:
  - Any camera
  - A CUDA compatible GPU
"""

import crappy

if __name__ == "__main__":
  graph = crappy.blocks.Grapher(('t(s)', 'x'), ('t(s)', 'y'), ('t(s)', 'r'),
                                length=50)

  correl = crappy.blocks.GPUCorrel(camera="Webcam",
      fields=['x', 'y', 'r'])  # Rigid body

  crappy.link(correl, graph)

  crappy.start()
