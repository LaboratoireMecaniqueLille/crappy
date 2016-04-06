#import ctypes
import numpy as np
#from numpy.ctypeslib import ndpointer
from matplotlib import pyplot as plt
import time
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
cam = cl.VideoCapture(0, "c:\\Users\\ECOLE\\fullareagray8.mcf" )
cam.set(cl.FG_HEIGHT, 2048)
cam.set(cl.FG_WIDTH, 2048)
cam.startAcq()
#cam = cl.Camera_new(boardNr, framespersec)
#cl.Camera_toString(cam)
#cl.Camera_init(cam, configFile)

def stop():
  cam.release()
  plt.close()

def start():
  try:
    ret, buf = cam.read()
    print "shape: ",buf.get('data').shape
    if(ret):
      plt.ion()
      plt.imshow(buf.get('data'),cmap='gray')
      plt.show()
    else:
      print "fail\n"
  except Exception as e:
    print "exception: ", e

def show():
  ret, buf = cam.read()
  if ret:
    plt.imshow(buf.get('data'),cmap='gray')
    plt.pause(0.001)
    plt.show()
  else:
    print "fail\n"

def loop(i):
  j=0
  while(j<i):
    show()
    j=j+1

def testPerf():
  t0 = time.time()
  i = 0
  while(i<200):
    ret, buf = cam.read()
    if(ret):
        plt.imshow(buf.get('data'),cmap='gray')
    else:
        print "fail"
    i = i+1
  t1 = time.time()
  print "FPS:", 200/(t1-t0)