# coding: utf-8
## @addtogroup sensor
# @{

##  @defgroup clSensor clSensor
# @{

## @file _clSensor.py
# @brief  To read data from cameralink cameras
#
# @author Victor Couty
# @version 0.1
# @date 22/02/2017
from __future__ import print_function,division

from ._meta import MasterCam
from . import clModule as cl
from time import time
import numpy as np

import Tkinter
import tkFileDialog

class CLCamera(MasterCam):
  """Cameralink camera sensor"""

  def __init__(self, numdevice = 0, config_file = None, camera_type = None):
    """Using the clModule, will open a cameraLink camera.
    If a config file is specified, it will be used to configure the camera
    If not set, it will be asked, unless set to False (or 0)
    Else, you must at least provide the camera type (eg: "FullAreaGray8")
    Using a config file is recommended over changing all settings manually
    """
    #print("config_file:",config_file)
    MasterCam.__init__(self)
    self.config_file = config_file
    self.camera_type = camera_type
    if config_file is None:
      root = Tkinter.Tk()
      root.withdraw()
      self.config_file = tkFileDialog.askopenfilename(parent=root)
      root.destroy()
    if self.camera_type is None and self.config_file:
      with open(self.config_file,'r') as f:
        r = f.readlines()
      r = filter(lambda s:s[:5]=="Typ='",r)
      if len(r) != 0:
        self.camera_type = r[0][5:-3]
    if self.camera_type is None:
      raise AttributeError("No camera type or valid config file specified!")
    self.name = "cl_camera"
    self.numdevice = numdevice
    self.add_setting("width",setter=self._set_w, getter=self._get_w)
    self.add_setting("height",setter=self._set_h, getter=self._get_h)
    self.add_setting("framespersec",setter=self._set_framespersec,getter=self._get_framespersec,limits=(1,200))

  def stopAcq(self):
    self.cap.stopAcq()

  def startAcq(self,*args):
    self.cap.startAcq(*args)

  def _set_framespersec(self,val):
    self.cap.set(cl.FG_FRAMESPERSEC,val)

  def _get_framespersec(self):
    return self.cap.get(cl.FG_FRAMESPERSEC)

  def _set_h(self,val):
    self.stopAcq()
    self.cap.set(cl.FG_HEIGHT,val)
    self.startAcq()

  def _set_w(self,val):
    self.stopAcq()
    self.cap.set(cl.FG_WIDTH,val)
    self.startAcq()

  def _get_h(self):
    return self.cap.get(cl.FG_HEIGHT)

  def _get_w(self):
    return self.cap.get(cl.FG_WIDTH)

  def open(self, **kwargs):
    """
    Opens the camera
    """
    if 'format' in kwargs:
      f = kwargs['format']
    else:
      if self.camera_type[-1] == '8':
        f = cl.FG_GRAY
      elif self.camera_type[-2:] == '16':
        f = cl.FG_GRAY16
      elif self.camera_type[-2:] == '24':
        f = cl.FG_COL24
      else:
        if self.config_file:
          with open(self.config_file,'r') as f:
            r = f.readlines()
          r = filter(lambda s:s[:10]=="FG_FORMAT=",r)
          if len(r) != 0:
            f = int(r[0].split['='][1])
          else:
            raise ValueError("Could not determine the format")
        else:
          raise ValueError("Could not determine the format")
    self.cap = cl.VideoCapture()
    self.cap.open(self.numdevice,self.camera_type,f)
    for k in kwargs:
      if not k in self.settings:
        raise AttributeError('Unexpected keyword: '+k)
    if self.config_file:
      self.cap.loadFile(self.config_file)
    self.set_all(**kwargs)
    # To make sure ROI is properly set up on first call
    for i in ['framespersec','height','width']:
      setattr(self,i,getattr(self,i))
    self.startAcq()
    self.configure()

  def configure(self):
    """Configure the frame grabber to trig the camera internally"""
    self.cap.set(cl.FG_TRIGGERMODE,1)
    self.cap.set(cl.FG_EXSYNCON,1)

  def get_image(self):
    t = time()
    r,f = self.cap.read()
    if not r:
      raise IOError("Could not read camera")
    return t,f

  def close(self):
    self.stopAcq()
    self.cap.release()
    self.cap = None
