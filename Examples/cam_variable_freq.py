# coding: utf-8

"""
This Example shows how to use the ``fps_label`` argument to change the
frequency of a camera during a test.

Required hardware:
  - Any camera
"""

import crappy

if __name__ == "__main__":
  camera = crappy.blocks.Camera(camera="Webcam",
                                verbose=True, fps_label='cmd')
  disp = crappy.blocks.Displayer(framerate=30)
  path = {'type': 'cyclic', 'value1': 30, 'condition1': 'delay=2',
          'value2': 5, 'condition2': 'delay=2', 'cycles': 0}
  freq_gen = crappy.blocks.Generator([path])
  crappy.link(freq_gen, camera)
  crappy.link(camera, disp)

  crappy.start()
