# coding: utf-8
##  @addtogroup sensor
# @{

##  @defgroup Webcam Webcam
# @{

## @file _webcamSensor.py
# @brief  Camera class for simple webcams, this class should inherit from CameraSensor
#
# @author Victor Couty
# @version 0.1
# @date 16/01/2017
from __future__ import print_function
from ._meta import MasterCam
import time
import numpy as np
import cv2



class Webcam(MasterCam):
  """
  Camera class for webcams, read using opencv
  """
  def __init__(self, numdevice=0):
    MasterCam.__init__(self)
    self.numdevice=numdevice
    self.name = "webcam"
    self.cap = None
    # No sliders for the camera: they usually only allow a few resolutions
    self.add_setting("width",640,self._set_w)
    self.add_setting("height",480,self._set_h)
    self.add_setting("channels",1,self._set_channels,{1:1,3:3})

  def _set_w(self,i):
    if self.cap:
      self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,i)
      r,f = self.cap.read()
      return r and f.shape[1] == i
    return False

  def _set_h(self,i):
    if self.cap:
      self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,i)
      r,f = self.cap.read()
      return r and f.shape[0] == i
    return False

  def _set_channels(self,i):
    return True

  def open(self,**kwargs):
    if self.cap:
      self.cap.release()
    self.cap = cv2.VideoCapture(self.numdevice)
    for k in kwargs:
      assert k in self.available_settings,str(self)+"Unexpected kwarg: "+str(k)
    self.set_all(**kwargs)

  def get_image(self):
    ret, frame = self.cap.read()
    if not ret:
      print("Error reading the camera")
      raise IOError
    if self.channels == 1:
      return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
      return frame#[:,:,[2,1,0]]

  def close(self):
    if self.cap:
      self.cap.release()
    self.cap = None
