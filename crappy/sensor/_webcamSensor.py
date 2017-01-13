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
from ._meta import cameraSensor
import time
import numpy as np
import cv2


class Webcam(cameraSensor.CameraSensor):
  """
  Camera class for webcams, simply read using opencv
  """

  def __init__(self, numdevice=0):
    self.name = "webcam"
    self.xoffset = self.yoffset = 0
    self.exposure = 1
    self.numdevice = numdevice
    self.cap = None
    #self.open()
    self.width = 640# self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    self.height = 480# self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    self.arguments = [('width',self.width),
                      ('height',self.height),
                      ('gain',1),
                      ('xoffset',0),
                      ('yoffset',0),
                      ('channels',1)]
    #self.close()

  def open(self):
    if not self.cap:
      print("opening cap")
      self.cap = cv2.VideoCapture(self.numdevice)

  def new(self, **kwargs):
    self.open()
    for arg,default in self.arguments:
      setattr(self,arg,kwargs.get(arg,default))
    inv = []
    for k in kwargs:
      if k not in map(lambda x:x[0],self.arguments):
        inv.append(k)
    if inv:
      print(self,"got invalid args:",*inv)
    assert self.channels in (1,3), "Incorrect number of channels: "+str(
                                                                self.channels)

  def get_image(self):
    ret, frame = self.cap.read()
    if not ret:
      print("Error reading the camera")
      raise IOError
    if self.channels == 1:
      return cv2.cvtColor(self.gain*frame, cv2.COLOR_BGR2GRAY)
    else:
      return self.gain*frame

  def close(self):
    if self.cap:
      print("Closing cap")
      self.cap.release()
      self.cap = None
