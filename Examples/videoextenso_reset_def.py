# coding: utf-8

"""
Example showing how to reset the reference length in the Videoextenso block.

Required hardware:
  - Any camera
"""

import crappy

if __name__ == "__main__":
  extenso = crappy.blocks.Video_extenso(camera="Webcam", show_image=True)

  graph_extenso = crappy.blocks.Grapher(('t(s)', 'Exx(%)'), ('t(s)', 'Eyy(%)'))
  crappy.link(extenso, graph_extenso)

  gui = crappy.blocks.GUI()
  crappy.link(gui, extenso)

  crappy.start()
