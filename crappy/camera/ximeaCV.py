# coding: utf-8
##  @addtogroup sensor
# @{

##  @defgroup XimeaCV XimeaCV
# @{

## @file _ximeaCV.py
# @brief  Camera class for Ximea camera, using opencv
#
# @author Victor Couty
# @version 0.1
# @date 02/12/2016

from __future__ import print_function

import time
import cv2

from .camera import Camera

#To check wether opencv can handle ximeas or not:
_ = cv2.CAP_PROP_XI_WIDTH # Will fail if the flag is not defined
del _

xi_format_dict = {'8 bits': 0, '10 bits': 1, '8 bits RAW': 5, '10 bits RAW': 6}
class XimeaCV(Camera):
  """
  Camera class for ximeas using openCV. It requires opencv 3.0 or higher,
   compiled with WITH_XIMEA flag
  """
  def __init__(self, numdevice=0):
    """
    Args:
        numdevice: Device number
    """
    Camera.__init__(self)
    self.numdevice = numdevice
    self.name = "XimeaCV"
    self.cap = None
    self.add_setting("width",self._get_w,self._set_w,(1,self._get_w))
    self.add_setting("height",self._get_h,self._set_h,(1,self._get_h))
    self.add_setting("xoffset",self._get_ox,self._set_ox,(0,self._get_w))
    self.add_setting("yoffset",self._get_oy,self._set_oy,(0,self._get_h))
    self.add_setting("exposure",self._get_exp,self._set_exp,(28,100000),10000)
    self.add_setting("gain",self._get_gain,self._set_gain,(0.,6.))
    self.add_setting("data_format",self._get_data_format,
                                   self._set_data_format,xi_format_dict)
    self.add_setting("AEAG",self._get_AEAG,self._set_AEAG,True,False)

  def _get_w(self):
    return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

  def _get_h(self):
    return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  def _get_ox(self):
    return int(self.cap.get(cv2.CAP_PROP_XI_OFFSET_X))

  def _get_oy(self):
    return int(self.cap.get(cv2.CAP_PROP_XI_OFFSET_Y))

  def _get_gain(self):
    return self.cap.get(cv2.CAP_PROP_XI_GAIN)

  def _get_exp(self):
    return int(self.cap.get(cv2.CAP_PROP_XI_EXPOSURE))

  def _get_AEAG(self):
    return bool(self.cap.get(cv2.CAP_PROP_XI_AEAG))

  def _get_data_format(self):
    return self.cap.get(cv2.CAP_PROP_XI_DATA_FORMAT)

  def _set_w(self,i):
    self.cap.set(cv2.CAP_PROP_XI_WIDTH,i)

  def _set_h(self,i):
    self.cap.set(cv2.CAP_PROP_XI_HEIGHT,i)

  def _set_ox(self,i):
    self.cap.set(cv2.CAP_PROP_XI_OFFSET_X,i)

  def _set_oy(self,i):
    self.cap.set(cv2.CAP_PROP_XI_OFFSET_Y,i)

  def _set_gain(self,i):
    self.cap.set(cv2.CAP_PROP_XI_GAIN,i)

  def _set_exp(self,i):
    self.cap.set(cv2.CAP_PROP_XI_EXPOSURE,i)

  def _set_AEAG(self,i):
    self.cap.set(cv2.CAP_PROP_XI_AEAG,int(bool(i)))

  def _set_data_format(self,i):
    self.cap.set(cv2.CAP_PROP_XI_DATA_FORMAT,i)

  def open(self,**kwargs):
    """
    Will actually open the camera, args will be set to default unless
    specified otherwise in kwargs
    """
    self.close()
    print("DEBUG nd=",self.numdevice)
    self.cap = cv2.VideoCapture(cv2.CAP_XIAPI+self.numdevice)

    for k in kwargs:
      assert k in self.available_settings,str(self)+"Unexpected kwarg: "+str(k)
    self.set_all(**kwargs)

  def reopen(self,**kwargs):
    """
    Will reopen the camera, args will be set to default unless
    specified otherwise in kwargs
    """
    self.close()
    self.cap = cv2.VideoCapture(cv2.CAP_XIAPI+self.numdevice)
    self.set_all(override=True,**kwargs)

  def get_image(self):
    """
    This method get a frame on the selected camera and return a ndarray

    If the camera breaks down, it reinitializes it, and tries again.

    Returns:
        frame from ximea device (ndarray height*width)
    """
    t = time.time()
    ret, frame = self.cap.read()
    if not ret:
      print("Error reading the camera!")
      print("Trying to reopen...")
      time.sleep(.5)
      print("Reopening with",self.settings_dict)
      self.reopen(**self.settings_dict)
      return self.get_image()
    return t,frame

  def close(self):
    """
    This method close properly the frame grabber.

    Returns:
        void return function.
    """
    if self.cap:
      self.cap.release()
    self.cap = None

  def __str__(self):
    """
    \__str\__ method to prints out main parameter.

    Returns:
        a formated string with the value of the main parameter.

    Example:
        camera = Ximea(numdevice=0)
        camera.new(exposure=10000, width=2048, height=2048)
        print camera

    these lines will print out:
         \code
         Exposure: 10000
         Numdevice: 0
         Width: 2048
         Height: 2048
         X offset: 0
         Y offset: 0
         \endcode
    """
    return " Exposure: {0} \n Numdevice: {1} \n Width: {2} \n Height: {3} " \
           "\n X offset: {4} \n Y offset: {5}".format(self.exposure,
                                  self.numdevice, self.width,
                                  self.height, self.xoffset, self.yoffset)
