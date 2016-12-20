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

from ._meta import cameraSensor
import time
import numpy as np
import cv2


class Webcam(cameraSensor.CameraSensor):
  """
  Camera class for webcams, simply read using opencv
  """

  def __init__(self, numdevice=0, framespersec=None, external_trigger=False):
    self.name = "webcam"
    self.cap = cv2.VideoCapture(numdevice)
    self.fps = framespersec
    self.external_trigger = external_trigger

  def new(self, width=None, height=None, gain=1, exposure=1, *args, **kwargs):
    print "CALLING NEW", args, kwargs
    self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    self.gain = gain
    self.exposure = exposure
    self.monochromatic = kwargs.get('monochromatic', True)

  def get_image(self):
    ret, frame = self.cap.read()
    if False and not ret:
      print "Error reading the camera"
      raise IOError
    if self.monochromatic:
      return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
      return frame

  def close(self):
    self.cap.release()
