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
# @date 02/12/2016
from __future__ import print_function
from ._meta import MasterCam
import time
import numpy as np
import cv2


class XimeaCV(MasterCam):
  """
  Camera class for ximeas using openCV (requires opencv 3.0 or higher)
  """

  def __init__(self, numdevice=0):
    MasterCam.__init__(self)
    self.name = "XimeaCV"
    self.numdevice = numdevice
    self.add_setting("width",2048,self._set_w,(1,2048))
    self.add_setting("height",2048,self._set_h,(1,2048))
    self.add_setting("xoffset",0,self._set_ox,(0,2044))
    self.add_setting("yoffset",0,self._set_oy,(0,2046))
    self.add_setting("exposure",10000,self._set_exp,(28,100000))
    self.add_setting("gain",1,self._set_gain,(0.,6.))
    self.add_setting("AEAG",False,self._set_AEAG,(False,True))
    self.cap = None
    #self.close()

  def _set_w(self,i):
    if self.cap:
      return self.cap.set(cv2.CAP_PROP_XI_WIDTH,i)
    return i % 4 == i and 0 < i <= 2048

  def _set_h(self,i):
    if self.cap:
      return self.cap.set(cv2.CAP_PROP_XI_HEIGHT,i)
    return i % 2 == i and 0 < i <= 2048

  def _set_ox(self,i):
    if self.cap:
      return self.cap.set(cv2.CAP_PROP_XI_OFFSET_X,i)
    return i % 4 == i and 0 < i <= 2048 - self.width

  def _set_oy(self,i):
    if self.cap:
      return self.cap.set(cv2.CAP_PROP_XI_OFFSET_Y,i)
    return i % 2 == i and 0 < i <= 2048 - self.height

  def _set_gain(self,i):
    if self.cap:
      return self.cap.set(cv2.CAP_PROP_XI_GAIN,i)
    return 0 <= i <= 6

  def _set_exp(self,i):
    if self.cap:
      return self.cap.set(cv2.CAP_PROP_XI_EXPOSURE,i)
    return i == int(i) and 28 <= i <= 1000000

  def _set_AEAG(self,i):
    if self.cap:
      return self.cap.set(cv2.CAP_PROP_XI_AEAG,int(bool(i)))
    return True

  def open(self):
    if self.cap:
      self.close()
    self.cap = cv2.VideoCapture(cv2.CAP_XIAPI+self.numdevice)
    self.set_all()

  def get_image(self):
    ret, frame = self.cap.read()
    if not ret:
      print("Error reading the camera")
      raise IOError
    return frame

  def close(self):
    if self.cap:
      self.cap.release()
    self.cap = None
