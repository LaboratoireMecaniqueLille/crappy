# coding: utf-8

"""
Very simple program that displays the output of a chosen camera.

Required hardware:
  - Any camera (select the desired camera in the terminal)
"""

import crappy

if __name__ == "__main__":
  cam_list = list(crappy.camera.camera_dict.keys())
  for i, c in enumerate(cam_list):
    print(i, c)
  r = int(input("What cam do you want to use ?> "))
  cam = cam_list[r]
  camera = crappy.blocks.Camera_parallel(camera=cam,
                                         verbose=True,
                                         display_images=True,
                                         displayer_framerate=20)

  crappy.start()
