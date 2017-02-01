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

from ._meta import MasterCam
import time
import cv2

#To check wether opencv can handle ximeas or not:
cv2.CAP_PROP_XI_WIDTH # Will fail if the flag is not defined

xi_format_dict = {'8 bits': 0, '10 bits': 1, '8 bits RAW': 5, '10 bits RAW': 6}
class XimeaCV(MasterCam):
  """
  Camera class for ximeas using openCV. It requires opencv 3.0 or higher,
   compiled with WITH_XIMEA flag
  """
  def __init__(self, numdevice=0):
    """
    Args:
        numdevice: Device number
    """
    MasterCam.__init__(self)
    self.numdevice = numdevice
    self.name = "XimeaCV"
    self.cap = None
    self.add_setting("width",2048,self._set_w,(1,4240))
    self.add_setting("height",2048,self._set_h,(1,2830))
    self.add_setting("xoffset",0,self._set_ox,(0,2044))
    self.add_setting("yoffset",0,self._set_oy,(0,2046))
    self.add_setting("exposure",10000,self._set_exp,(28,100000))
    self.add_setting("gain",1,self._set_gain,(0.,6.))
    self.add_setting("data_format",0,self._set_data_format,xi_format_dict)
    self.add_setting("AEAG",False,self._set_AEAG,True)

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

  def _set_data_format(self,i):
    if self.cap:
      return self.cap.set(cv2.CAP_PROP_XI_DATA_FORMAT,i)
    return abs(int(i)) == i and i <=6

  def open(self,**kwargs):
    """
    Will actually open the camera, args will be set to default unless 
    specified otherwise in kwargs
    """
    if self.cap:
      self.close()
    self.cap = cv2.VideoCapture(cv2.CAP_XIAPI+self.numdevice)

    for k in kwargs:
      assert k in self.available_settings,str(self)+"Unexpected kwarg: "+str(k)
    self.set_all(**kwargs)

  def get_image(self):
    """
    This method get a frame on the selected camera and return a ndarray

    If the camera breaks down, it reinitializes it, and tries again.

    Returns:
        frame from ximea device (ndarray height*width)
    """
    ret, frame = self.cap.read()
    if not ret:
      print("Error reading the camera!")
      print("Trying to reopen...")
      time.sleep(1)
      self.open(**self.settings_dict)
      ret, frame = self.cap.read()
      if not ret:
        raise IOError("Error reading Ximea camera!")
      print("Phew, I got it! (Some frames were lost, thought...)")
    return frame

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
