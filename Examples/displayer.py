# coding: utf-8

"""
Very simple program that displays the output of a chosen camera.

Required hardware:
  - Any camera (select the desired camera in the terminal)
"""

import crappy

if __name__ == "__main__":
  cam_list = list(crappy.camera.MetaCam.classes.keys())
  cam_list.remove("Camera")
  for i, c in enumerate(cam_list):
    print(i, c)
  r = int(input("What cam do you want to use ?> "))
  cam = cam_list[r]
  camera = crappy.blocks.Camera(camera=cam, verbose=True)

  disp = crappy.blocks.Displayer(framerate=20)

  crappy.link(camera, disp)

  crappy.start()
