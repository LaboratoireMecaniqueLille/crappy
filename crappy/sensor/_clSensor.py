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
from . import clModule
from time import time
import numpy as np
import Tkinter
import tkFileDialog

class CLCamera(MasterCam):
  """Cameralink camera sensor"""

  def __init__(self, numdevice = 0, config_file = None, camera_type = None):
    """Using the clModule, will open a cameraLink camera.
    If a config file is specified, it will be used to configure the camera
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
    if self.camera_type is None:
      with open(config_file,'r') as f:
        r = f.readlines()
      r = filter(lambda s:s[:5]=="Typ='",r)
      if len(r) != 0:
        self.camera_type = r[0][5:-3]
    if camera_type is None:
      raise AttributeError("No camera type or valid config file specified!")
    self.name = "cl_camera"
    self.numdevice = numdevice
    self.add_setting("width",setter=self._set_w, getter=self._get_w)
    self.add_setting("height",setter=self._set_h, getter=self._get_h)
    self.add_setting("framespersec",setter=self._set_framespersec,getter=self._get_framespersec,limits=(1,200))
    self.acqStarted = False

  def stopAcq(self):
    if self.acqStarted:
      self.cap.stopAcq()
    self.acqStarted = False

  def startAcq(self):
    if not self.acqStarted:
      self.cap.startAcq()
    self.acqStarted = True

  def _set_framespersec(self,val):
    self.cap.set(clModule.FG_FRAMESPERSEC,val)

  def _get_framespersec(self):
    return self.cap.get(clModule.FG_FRAMESPERSEC)

  def _set_h(self,val):
    self.stopAcq()
    self.cap.set(clModule.FG_HEIGHT,val)
    self.startAcq()

  def _set_w(self,val):
    self.stopAcq()
    self.cap.set(clModule.FG_WIDTH,val)
    self.startAcq()

  def _get_h(self):
    return self.cap.get(clModule.FG_HEIGHT)

  def _get_w(self):
    return self.cap.get(clModule.FG_WIDTH)

  def open(self, **kwargs):
    """
    Opens the camera
    """
    self.cap = clModule.VideoCapture(self.numdevice,self.config_file,self.camera_type)
    for k in kwargs:
      if not k in self.settings:
        raise AttributeError('Unexpected keyword: '+k)
    self.set_all(**kwargs)
    # To make sure ROI is properly set up on first call
    CLCamera._set_w(self,self.width)
    CLCamera._set_h(self,self.height)
    self.startAcq()

  def get_image(self):
    t = time()
    r,f = self.cap.read()
    if not r:
      raise IOError("Could not read camera")
    #print('DEBUG shape',f['data'].shape)
    return t,f['data']

  def close(self):
    self.stopAcq()
    self.cap.release()
    self.cap = None
