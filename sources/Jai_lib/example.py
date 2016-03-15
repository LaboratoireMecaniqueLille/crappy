#import ctypes
import numpy as np
#from numpy.ctypeslib import ndpointer
from matplotlib import pyplot as plt
#cl = ctypes.CDLL('/home/jai5000/Code/crappy/sources/Jai_lib/cameraLinkModule.so')
#configFile= ctypes.c_char_p("/home/jai5000/Code/crappy/sources/Jai_lib/config.mcf")
#boardNr = 0	
#exposure = 4000
#width=2560
#height=2048
#xoffset=0
#yoffset=0
#FPS=80
#framespersec=ctypes.c_double(FPS)
from crappy.sensor import clModule as cl
cam = cl.VideoCapture(1, '/home/jai5000/Documents/mediumgray.mcf')
cam.startAcq()
#cam = cl.Camera_new(boardNr, framespersec)
#cl.Camera_toString(cam)
#cl.Camera_init(cam, configFile)

def stop():
  cam.release() 
  #cl.Camera_close(cam)
  plt.close()
  
try:
  #cl.Camera_Buffer.restype = ndpointer(dtype=np.uintp, shape=(width,height))
  #buf = cl.Camera_Buffer(cam)
  #plt.imshow(buf,cmap='gray')
  ret, buf = cam.read()
  plt.imshow(buf.get('data'))
  plt.ion()
  plt.show()
except Exception as e:
  print "exception: ", e

def show():
  #buf = cl.Camera_Buffer(cam)
  #plt.imshow(buf,cmap='gray')
  ret, buf = cam.read()
  plt.imshow(buf.get('data'),cmap='gray')
  plt.draw()
