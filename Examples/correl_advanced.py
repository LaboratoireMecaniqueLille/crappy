# coding: utf-8

"""
Shows a more advanced way of using the GPUCorrel block.

Required hardware:
  - Any camera
  - A CUDA compatible GPU
"""

import numpy as np

import crappy


if __name__ == "__main__":
  CAMERA = "Webcam"

  x = 2048
  y = 2048

  mask = np.empty((y, x), np.float32)
  r = min(x, y) / 2.
  r -= .1 * r
  r2 = r * r

  # Generating a circular weighted mask: the further from the center,
  # the lower the weight will be
  # Because why not ?
  for i in range(x):
    for j in range(y):
      mask[j, i] = max(0., 1 - ((i - x / 2) ** 2 + (j - x / 2) ** 2) /
                       (min(x, y) / 2.1) ** 2)

  # Generating your own displacement field:
  # It is simply a tuple of numpy arrays: one for the disp along X, one for Y
  # For example, let's make the fields corresponding to X translation:
  # Note: we could have used the default 'x' in this case,
  # this is just an example to show how to use custom fields
  myX = (np.ones((y, x), np.float32), np.zeros((y, x), np.float32))

  graph = crappy.blocks.Grapher(('t(s)', 'x'), ('t(s)', 'y'), ('t(s)', 'r'),
                                length=50)
  graphRes = crappy.blocks.Grapher(('t(s)', 'res'), length=50)
  graphLinDef = crappy.blocks.Grapher(('t(s)', 'Exx'), ('t(s)', 'Exy'))
  graphQuadDef = crappy.blocks.Grapher(('t(s)', 'Ux2'), ('t(s)', 'Vy2'))
  # Creating the correl block
  correl = crappy.blocks.GPUCorrel(camera=CAMERA,
                                   fields=[
                                     myX, 'y', 'r',  # Rigid body
                                     'exx', 'eyy', 'exy',   # Linear def
                                     'uxx', 'uyy', 'uxy',   # Quadratic def (x)
                                     'vxx', 'vyy', 'vxy'],  # Quadratic def (y)
                                   verbose=2,  # To print info
                                   show_diff=True,  # Display the residual
                                   # (slow!)
                                   drop=False,  # Disable data picker
                                   mask=mask,
                                   levels=4,  # Reduce the number of levels
                                   iterations=3,  # and of iteration
                                   resampling_factor=2.5,  # aggressive
                                   # resampling
                                   labels=(  # Needed to name our custom field
                                     'x', 'y', 'r', 'Exx', 'Eyy', 'Exy',
                                     'Ux2', 'Uy2', 'Uxy',
                                     'Vx2', 'Vy2', 'Vxy'),
                                   mul=3.2,  # Scalar to multiply the direction
                                   res=True)  # Ask to return the residual

  # Link compacter to main graph
  crappy.link(correl, graph)
  crappy.link(correl, graphRes)
  crappy.link(correl, graphLinDef)
  crappy.link(correl, graphQuadDef)

  crappy.start()
