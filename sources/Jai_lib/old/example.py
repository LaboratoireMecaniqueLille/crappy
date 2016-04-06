import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from matplotlib import pyplot as plt
cl = ctypes.CDLL('cameraLinkModule.dll')
configFile= ctypes.c_char_p("testconfig.mcf")
boardNr = 0	
exposure = 4000
width=2048
height=2048
xoffset=0
yoffset=0
FPS=80
framespersec=ctypes.c_double(FPS)

cam = cl.Camera_new(boardNr, exposure, width, height, xoffset, yoffset, FPS)
cl.Camera_toString(cam)
cl.Camera_init(cam, configFile)

def stop(camera):
  cl.Camera_close(camera)
  plt.close()
  
try:
  cl.Camera_Buffer.restype = ndpointer(dtype=np.uint8, shape=(width,height))
  buf = cl.Camera_Buffer(cam)
  plt.imshow(buf,cmap='gray')
  plt.ion()
  plt.show()
except Exception as e:
  print "exception: ", e

def _show():
  buf = cl.Camera_Buffer(cam)
  plt.imshow(buf,cmap='gray')
  plt.draw()
