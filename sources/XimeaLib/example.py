import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from matplotlib import pyplot as plt
lib = ctypes.cdll.LoadLibrary('ximeaModule.dll')
CAP_PROP_XI_OFFSET_X    = 402
CAP_PROP_XI_OFFSET_Y    = 403
CAP_PROP_FRAME_WIDTH    =3
CAP_PROP_FRAME_HEIGHT   =4


cam = lib.VideoCapture(0)
lib.get(cam, CAP_PROP_FRAME_WIDTH)
lib.get(cam, CAP_PROP_FRAME_HEIGHT)

width = 2048
height = 2048
lib.set(cam, CAP_PROP_FRAME_WIDTH, width)
lib.set(cam, CAP_PROP_FRAME_HEIGHT, height)

lib.get(cam, CAP_PROP_FRAME_WIDTH)
lib.get(cam, CAP_PROP_FRAME_HEIGHT)


def stop(camera):
  cl.Camera_close(camera)
  plt.close()
  
try:
  lib.read.restype = ndpointer(dtype=np.uint8, shape=(width,height))
  buf = lib.read(cam)
  plt.imshow(buf,cmap='gray')
  plt.ion()
  plt.show()
except Exception as e:
  print "exception: ", e

def show():
  buf = lib.read(cam)
  plt.imshow(buf,cmap='gray')
  plt.draw()
