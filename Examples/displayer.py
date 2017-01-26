#coding: utf-8

import crappy
import time

"""
Very simple program that displays the output of a choosen camera
"""
if __name__ == "__main__":
  cam_list = crappy.sensor._meta.MetaCam.classes.keys()
  cam_list.remove("MasterCam")
  for i,c in enumerate(cam_list):
    print i,c
  r = int(raw_input("What cam do you want to use ?> "))
  cam = cam_list[r]
  camera = crappy.blocks.StreamerCamera(camera=cam,show_fps=True)

  disp = crappy.blocks.CameraDisplayer(framerate=20)

  crappy.link(camera,disp)

  crappy.start()
