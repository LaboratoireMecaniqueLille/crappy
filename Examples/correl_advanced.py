import crappy
#import crappy
from time import time
import numpy as np


x=2048
y=2048



mask = np.empty((y,x),np.float32)
r = min(x,y)/2.
r -= .1*r
r2 = r*r

# Generating a circular weighted mask: the further from the center, 
# the lower the weight will be
# Because why not ?
for i in range(x):
  for j in range(y):
    mask[j,i] = max(0,1-((i-x/2)**2+(j-x/2)**2)/(min(x,y)/2.1)**2)

# Generating your own displacement field:
# It is simply a tuple of numpy arrays: one for the disp along X, one for Y
# For example, let's make the fields corresponding to X translation:
# Note: we could have used the default 'x' in this case, 
# this is just an example to show how to use custom fields
myX = (np.ones((y,x),np.float32), np.zeros((y,x),np.float32))


camera = crappy.blocks.StreamerCamera("Ximea", numdevice=0, freq=20,
                                save=False,save_directory="/home/vic/outTest/",
                                xoffset=0, yoffset=0, width=x, height=y)
graph = crappy.blocks.Grapher(('t','x'),('t','y'),('t','r'),length=50)
graphRes = crappy.blocks.Grapher(('t','res'),length=50)
graphLinDef = crappy.blocks.Grapher(('t','Exx'),('t','Exy'))
graphQuadDef = crappy.blocks.Grapher(('t','Ux2'),('t','Vy2'))
# Creating the correl block
correl = crappy.blocks.Correl((y,x),fields=[myX,'y','r', # Rigid body
                              'exx','eyy','exy', # Linear def
                              'uxx','uyy','uxy', # Quadratic def (x)
                              'vxx','vyy','vxy'],# Quadratic def (y)
                              verbose=2, #To print info
                              show_diff=True, # Display the residual (slow!)
                              drop=False, # Disable datapicker
                              mask=mask,
                              levels=4, # Reduce the number of levels
                              iterations=3, # and of iteration
                              resampling_factor=2.5, # agressive resampling
                              labels=( #Needed to name our custom field
                              'x','y','r','Exx','Eyy','Exy',
                              'Ux2','Uy2','Uxy',
                              'Vx2','Vy2','Vxy'),
                              mul=3.2, # Scalar to multiply the direction
                              res=True)# Ask to return the residual

compacter = crappy.blocks.Compacter(3)

# Link camera to correl block
lCam2Correl = crappy.links.Link()
camera.add_output(lCam2Correl)
correl.add_input(lCam2Correl)
# Link Correl block to compacter
lCorrel2Comp = crappy.links.Link()
correl.add_output(lCorrel2Comp)
compacter.add_input(lCorrel2Comp)
# Link compacter to main graph
lComp2Graph = crappy.links.Link()
compacter.add_output(lComp2Graph)
graph.add_input(lComp2Graph)
# Link compacter to Residual graph
lComp2GraphRes = crappy.links.Link()
compacter.add_output(lComp2GraphRes)
graphRes.add_input(lComp2GraphRes)
# Link compacter to Linear deformations graph
lComp2GraphLinDef = crappy.links.Link()
compacter.add_output(lComp2GraphLinDef)
graphLinDef.add_input(lComp2GraphLinDef)
# Link compacter to quadratic deformations graph
lComp2GraphQuadDef = crappy.links.Link()
compacter.add_output(lComp2GraphQuadDef)
graphQuadDef.add_input(lComp2GraphQuadDef)

correl.init() # < -- DO NOT FORGET THIS !
print "[Main program] Ready to go !"

t0 = time()
for i in crappy.blocks.MasterBlock.instances:
  i.t0 = t0
try:
  for i in crappy.blocks.MasterBlock.instances:
    i.start()
except Exception as e:
  print "[Main program] Caught an exception:",e
  for i in crappy.blocks.MasterBlock.instances:
    print i
    i.stop()
  raise
