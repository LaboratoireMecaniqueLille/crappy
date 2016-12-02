#coding: utf-8

import crappy
import time

"""
Very simple program that 
"""
camera = crappy.blocks.StreamerCamera("Ximea", numdevice=0, freq=20, save=False,save_directory="CHANGEME",xoffset=0, yoffset=0, width=2048, height=2048)

disp = crappy.blocks.CameraDisplayer(framerate=10)

link = crappy.links.Link()

camera.add_output(link)
disp.add_input(link)

t0=time.time()
for instance in crappy.blocks._meta.MasterBlock.instances:
  instance.t0 = t0

for instance in crappy.blocks._meta.MasterBlock.instances:
  instance.start()

while True:
  time.sleep(1)
  print "Elapsed:",time.time()-t0
