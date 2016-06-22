import crappy2 as crappy
#import crappy
from time import time
import numpy as np


x=2048
y=2048


ones = np.ones((x,y),dtype=np.float32)
zeros = np.zeros((x,y),dtype=np.float32)

mvX = (ones,zeros)
mvY = (zeros,ones)

sq = .5**.5

Zoom = np.meshgrid(np.arange(-sq,sq,2*sq/x,dtype=np.float32),np.arange(-sq,sq,2*sq/y,dtype=np.float32))

Zoom = (Zoom[0].astype(np.float32),Zoom[1].astype(np.float32))
Rot = (Zoom[1],-Zoom[0])




camera = crappy.blocks.StreamerCamera("Ximea", numdevice=0, freq=30, save=False,save_directory="/home/vic/outTest/",xoffset=0, yoffset=0, width=x, height=y)
graph = crappy.blocks.Grapher("dynamic",('t','x'),('t','y'),('t','R'))
correl = crappy.blocks.Correl((x,y),fields=(mvX,mvY,Rot),verbose=2,levels=6,labels=('x','y','R'))
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

"""
display = crappy.blocks.CameraDisplayer()
lCam2Disp = crappy.links.Link()
camera.add_output(lCam2Disp)
display.add_input(lCam2Disp)
"""


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
