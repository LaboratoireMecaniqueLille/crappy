import crappy2 as crappy
#import crappy
from time import time
import numpy as np


x=2048
y=2048




camera = crappy.blocks.StreamerCamera("Ximea", numdevice=0, freq=50, save=False,save_directory="/home/vic/outTest/",xoffset=0, yoffset=0, width=x, height=y)
graph = crappy.blocks.Grapher(('t','x'),('t','y'),('t','r'),length=30)
correl = crappy.blocks.Correl((y,x),fields=['x','y','r'],verbose=2,show_diff=True)
compacter = crappy.blocks.Compacter(5)

lCam2Correl = crappy.links.Link()
camera.add_output(lCam2Correl)
correl.add_input(lCam2Correl)

lCorrel2Comp = crappy.links.Link()
correl.add_output(lCorrel2Comp)
compacter.add_input(lCorrel2Comp)

lComp2Graph = crappy.links.Link()
compacter.add_output(lComp2Graph)
graph.add_input(lComp2Graph)
#"""
display = crappy.blocks.CameraDisplayer(framerate=10)
lCam2Disp = crappy.links.Link()
camera.add_output(lCam2Disp)
display.add_input(lCam2Disp)
#"""

correl.init()
print "Ready to go !"

t0 = time()
for i in crappy.blocks.MasterBlock.instances:
  i.t0 = t0
try:
  for i in crappy.blocks.MasterBlock.instances:
    i.start()
except Exception as e:
  print "Main program caught an exception:",e
  for i in crappy.blocks.MasterBlock.instances:
    print i
    i.stop()
  raise
