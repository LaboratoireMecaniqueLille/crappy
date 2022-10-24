# coding: utf-8

"""
Example showing how to use the DISVE block.

It performs DIC on the specified patches, and returns the motion along `x` and
`y` in pixels. This example demonstrates the processing to measure strain using
this method.

!! Important !!
  The w and h values may need to be adjusted to the actual size of your image.
  Otherwise an error may be raised.

  It was not possible to set them automatically here while still keeping the
  script short and easily understandable.

Required hardware:
  - Any camera
"""

import crappy

if __name__ == "__main__":
  w, h = 640, 480
  ps = 100  # patch size (x and y)
  m = 100  # Margin

  # Patches are defined as such: (y, x, height, width)
  # x and y being the coordinates to the upper-left corner
  patches = [
      (m, w // 2 - ps // 2, ps, ps),  # Top
      (h // 2 - ps // 2, w - ps - m, ps, ps),  # Right
      (h - ps - m, w // 2 - ps // 2, ps, ps),  # Bottom
      (h // 2 - ps // 2, m, ps, ps)]  # Left

  ve = crappy.blocks.DISVE('Webcam', patches, verbose=True,
                           display_images=True)
  graphy = crappy.blocks.Grapher(('t(s)', 'p0y'), ('t(s)', 'p2y'))
  graphx = crappy.blocks.Grapher(('t(s)', 'p1x'), ('t(s)', 'p3x'))

  crappy.link(ve, graphx)
  crappy.link(ve, graphy)

  def compute_strain(d):
    d['Exx(%)'] = (d['p3x'] - d['p1x']) / (w - ps) * 100
    d['Eyy(%)'] = (d['p0y'] - d['p2y']) / (h - ps) * 100
    return d

  graphstrain = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))
  crappy.link(ve, graphstrain, modifier=compute_strain)
  crappy.start()
