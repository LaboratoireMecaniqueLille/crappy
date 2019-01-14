#coding: utf-8

import crappy

"""
Very simple program that displays the output of a choosen camera
"""
if __name__ == "__main__":
  camera = crappy.blocks.Camera(camera="Webcam",
      verbose=True,fps_label='cmd')
  disp = crappy.blocks.Displayer(framerate=30)
  path = {'type':'cyclic','value1':30,'condition1':'delay=2',
                        'value2':5,'condition2':'delay=2','cycles':0}
  freq_gen = crappy.blocks.Generator([path])
  crappy.link(freq_gen,camera)
  crappy.link(camera,disp)

  crappy.start()
