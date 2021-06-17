# coding: utf-8

"""
Very simple test demonstrating the video-extensometry.

To test it, one can draw markers on a paper an move it closer or further to the
camera. This should induce a strain on both `x` and `y` directions.

Required hardware:
  - A camera
"""

import crappy

if __name__ == '__main__':
  graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))

  extenso = crappy.blocks.Video_extenso(camera="Webcam",
      end=True, show_image=True, white_spots=False)

  crappy.link(extenso, graph_extenso)
  crappy.start()
