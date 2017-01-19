#coding: utf-8

import crappy
import time

"""
Very simple program that displays the output of a camera
"""
class Print_val(crappy.links.Condition):
  def evaluate(self,data):
    print("GOT:",type(data))
    try:
      print("shape:",data.shape)
    except AttributeError:
      pass
    return data

if __name__ == "__main__":
  #camera = crappy.blocks.StreamerCamera(camera="Ximea", freq=20, save=False)
  #camera = crappy.blocks.StreamerCamera(camera="Ximea", save=False, show_fps=True)
  #camera = crappy.blocks.FakeCamera(512,512)
  cam_list = crappy.sensor._meta.MetaCam.classes.keys()
  cam_list.remove("MasterCam")
  #camera = crappy.blocks.StreamerCamera(camera="Fake_camera")
  for i,c in enumerate(cam_list):
    print i,c
  r = int(raw_input("What cam do you want to use ?> "))
  cam = cam_list[r]
  camera = crappy.blocks.StreamerCamera(camera=cam)

  disp = crappy.blocks.CameraDisplayer(framerate=None)
  #disp = crappy.blocks.Sink()

  crappy.link(camera,disp)
  #crappy.links.Link(camera,disp,condition=Print_val())

  crappy.start()
