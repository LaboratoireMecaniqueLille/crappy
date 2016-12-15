#coding: utf-8

import crappy
import time

"""
Very simple program that displays the output of a camera
"""
if __name__ == "__main__":
  #camera = crappy.blocks.StreamerCamera("Ximea", numdevice=0, freq=20, save=False,save_directory="CHANGEME",xoffset=0, yoffset=0, width=2048, height=2048)
#camera = crappy.blocks.FakeCamera(512,512)
  camera = crappy.blocks.StreamerCamera("Webcam", save=True, save_directory='/home/francois/Images/')

  disp = crappy.blocks.CameraDisplayer(framerate=9999999)

#link = crappy.links.Link()
  crappy.links.Link(camera,disp)

#	camera.add_output(link)
#	disp.add_input(link)

#	t0=time.time()
#	for instance in crappy.blocks._masterblock.MasterBlock.instances:
#	  instance.t0 = t0
#
#	for instance in crappy.blocks._masterblock.MasterBlock.instances:
#	  instance.start()

  crappy.start()
