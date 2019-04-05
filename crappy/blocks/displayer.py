# coding: utf-8

import numpy as np
from .masterblock import MasterBlock
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import ImageTk,Image


class Displayer(MasterBlock):
  """
  Simple image displayer using openCV or Matplotlib

  It can be paired with StreamerCamera
  Use cv=False to use the old, inefficient and deprecated version
  NOTE: You need to use one displayer block per window
  (in other words, you can only attach one input to the diplayer)
  """
  def __init__(self, framerate=5, backend='cv', title='Displayer'):
    MasterBlock.__init__(self)
    self.niceness = 10
    if framerate is None:
      self.delay = 0
    else:
      self.delay = 1. / framerate  # Framerate (fps)
    self.title = title
    if backend.lower() in ['cv','opencv']:
      self.loop = self.loop_cv
      self.begin = self.begin_cv
      self.finish = self.finish_cv
    elif backend.lower() in ['matplotlib','mpl']:
      self.loop = self.loop_mpl
      self.begin = self.begin_mpl
      self.finish = self.finish_mpl
    elif backend.lower() in ['tk','tkinter']:
      self.loop = self.loop_tk
      self.begin = self.begin_tk
      self.finish = self.finish_tk
    else:
      raise AttributeError("Unknown backend: "+str(backend))

  # Matplotlib
  def begin_mpl(self):
    plt.ion()
    fig = plt.figure()
    fig.add_subplot(111)

  def cast_8bits(self,f):
    m = f.max()
    i = 0
    while m >= 2**(8+i):
      i+=1
    return (f/(2**i)).astype(np.uint8)

  def loop_mpl(self):
    data = self.inputs[0].recv_delay(self.delay)['frame'][-1]
    plt.imshow(data, cmap='gray')
    plt.pause(0.001)
    plt.show()

  def finish_mpl(self):
    plt.close('all')

  # OpenCV
  def begin_cv(self):
    try:
      flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    # WINDOW_KEEPRATIO is not implemented in all opencv versions...
    except AttributeError:
      flags = cv2.WINDOW_NORMAL
    cv2.namedWindow(self.title, flags)

  def loop_cv(self):
    data = self.inputs[0].recv_delay(self.delay)['frame'][-1]
    if data.dtype != np.uint8:
      if data.max() >= 256:
        data = self.cast_8bits(data)
      else:
        data = data.astype(np.uint8)
    cv2.imshow(self.title,data)
    cv2.waitKey(1)

  def finish_cv(self):
    cv2.destroyAllWindows()

  # TKinter
  def resize(self,img):
    return cv2.resize(img,(self.w,self.h))

  def begin_tk(self):
    self.root = tk.Tk()
    self.imglabel = tk.Label(self.root)
    self.imglabel.pack()
    self.h = 480
    self.w = 640

  def loop_tk(self):
    data = self.inputs[0].recv_delay(self.delay)['frame'][-1]
    if data.dtype != np.uint8:
      if data.max() >= 256:
        data = self.cast_8bits(data)
      else:
        data = data.astype(np.uint8)
    cimg = ImageTk.PhotoImage(Image.fromarray(self.resize(data)))
    self.imglabel.configure(image=cimg)
    self.root.update()

  def finish_tk(self):
    self.root.destroy()
