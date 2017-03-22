#coding: utf-8

import crappy

"""
Very simple program that displays the output of a choosen camera
"""
if __name__ == "__main__":
  cam_list = crappy.camera.MetaCam.classes.keys()
  cam_list.remove("Camera")
  for i,c in enumerate(cam_list):
    print i,c
  r = int(raw_input("What cam do you want to use ?> "))
  cam = cam_list[r]
  camera = crappy.blocks.Camera(camera=cam,show_fps=True)

  disp = crappy.blocks.Displayer(framerate=20)

  crappy.link(camera,disp)

  crappy.start()
