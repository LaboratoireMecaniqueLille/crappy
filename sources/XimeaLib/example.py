import time
import crappy
import crappy.sensor.ximeaModule as xi
import numpy as np
from matplotlib import pyplot as plt
import cv2

ximea = xi.VideoCapture(0)
ximea.set(xi.CAP_PROP_XI_DATA_FORMAT,0) #0=8 bits, 1=16(10)bits, 5=8bits RAW, 6=16(10)bits RAW  
ximea.set(xi.CAP_PROP_XI_AEAG,0)#auto gain auto exposure
ximea.set(xi.CAP_PROP_FRAME_HEIGHT,2048)
ximea.set(xi.CAP_PROP_FRAME_WIDTH,2048)
ximea.set(xi.CAP_PROP_XI_OFFSET_Y,0)
ximea.set(xi.CAP_PROP_XI_OFFSET_X,0)

def stop():
  ximea.release()
  plt.close()

def start():
  try:
    ximea.addTrigger(10000000, True)
    ret, buf = ximea.read()
    print time.time()
    if(ret):
      plt.ion()
      plt.imshow(buf.get('data'),cmap='gray')
      plt.show()
    else:
      print "fail\n"
  except Exception as e:
    print "exception: ", e

def show():
  ret, buf = ximea.read()
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
    ret, buf = ximea.read()
    if(ret):
        plt.imshow(buf.get('data'),cmap='gray')
    else:
        print "fail"
    i = i+1
  t1 = time.time()
  print "FPS:", 200/(t1-t0)