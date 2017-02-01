# coding: utf-8
##  @addtogroup sensor
# @{

##  @defgroup Ximea Ximea
# @{

## @file _ximeaSensor.py
# @brief  Camera class for ximea devices, this class should inherit from MasterCam
#
# @author Robin Siemiatkowski, Victor Couty
# @version 0.2
# @date 26/01/2017

from __future__ import print_function

from time import sleep

from ._meta import MasterCam
import ximeaModule as xi


xi_format_dict = {'8 bits': 0, '10 bits': 1, '8 bits RAW': 5, '10 bits RAW': 6}

class Ximea(MasterCam):
  """
  Camera class for ximea devices, this class should inherit from MasterCam
  This class cannot go beyond 30~35 fps depending on your hardware, but does
  not require openCV3 with Ximea flags
  """

  def __init__(self,numdevice=0):
    """
    Args:
        numdevice: Device number
    """
    MasterCam.__init__(self)
    self.numdevice = numdevice
    self.name = "Ximea"
    self.ximea = None
    self.add_setting("width",2048,self._set_w,(1,2048))
    self.add_setting("height",2048,self._set_h,(1,2048))
    self.add_setting("xoffset",0,self._set_ox,(0,2044))
    self.add_setting("yoffset",0,self._set_oy,(0,2046))
    self.add_setting("exposure",10000,self._set_exp,(28,100000))
    self.add_setting("gain",1,self._set_gain,(0.,6.))
    self.add_setting("data_format",0,self._set_data_format,xi_format_dict)
    self.add_setting("AEAG",False,self._set_AEAG,True)
    self.add_setting("External Trigger",False,self._set_ext_trig)

  def open(self, **kwargs):
    """
    Will actually open the camera, args will be set to default unless 
    specified otherwise in kwargs
    """
    self.close() #If it was already open (won't do anything if cam is not open)

    if type(self.numdevice) == str:
      # open the ximea device Ximea devices start at 1100. 1100 => device 0, 1101 => device 1
      self.ximea = xi.VideoCapture(device_path=self.numdevice)
    else:
      self.ximea = xi.VideoCapture(self.numdevice)
    # Will apply all the settings to default or specified value
    self.set_all(**kwargs) 

  def _set_w(self,i):
    if self.ximea:
      return self.ximea.set(xi.CAP_PROP_FRAME_WIDTH,i)
    return i % 4 == i and 0 < i <= 2048

  def _set_h(self,i):
    if self.ximea:
      return self.ximea.set(xi.CAP_PROP_FRAME_HEIGHT,i)
    return i % 2 == i and 0 < i <= 2048

  def _set_ox(self,i):
    if self.ximea:
      return self.ximea.set(xi.CAP_PROP_XI_OFFSET_X,i)
    return i % 4 == i and 0 < i <= 2048 - self.width

  def _set_oy(self,i):
    if self.ximea:
      return self.ximea.set(xi.CAP_PROP_XI_OFFSET_Y,i)
    return i % 2 == i and 0 < i <= 2048 - self.height

  def _set_gain(self,i):
    if self.ximea:
      return self.ximea.set(xi.CAP_PROP_GAIN,i)
    return 0 <= i <= 6

  def _set_exp(self,i):
    if self.ximea:
      return self.ximea.set(xi.CAP_PROP_EXPOSURE,i)
    return i == int(i) and 28 <= i <= 1000000

  def _set_ext_trig(self,i):
    if self.ximea:
      self.ximea.addTrigger(1000000, int(bool(i)))
    return True

  def _set_data_format(self,i):
    # 0=8 bits, 1=16(10)bits, 5=8bits RAW, 6=16(10)bits RAW
    if self.ximea:
      if i == 1 or i == 6:  
      # increase the FPS in 10 bits
        self.ximea.set(xi.CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH, 10)
        self.ximea.set(xi.CAP_PROP_XI_DATA_PACKING, 1)
      return self.ximea.set(xi.CAP_PROP_XI_DATA_FORMAT, i)  
    return abs(int(i)) == i and i <=6

  def _set_AEAG(self,i):
    if self.ximea:
      return self.ximea.set(xi.CAP_PROP_XI_AEAG,int(bool(i)))
    return True

  def get_image(self):
    """
    This method get a frame on the selected camera and return a ndarray

    If the camera breaks down, it reinitializes it, and tries again.

    Returns:
        frame from ximea device (ndarray height*width)
    """
    ret, frame = self.ximea.read()
    try:
      if ret:
        return frame.get('data')

      print("restarting camera...")
      sleep(2)
      self.open(**self.settings_dict)
      return self.get_image()
    except UnboundLocalError:  
      # if ret doesn't exist, because of KeyboardInterrupt
      print("ximea quitting, probably because of KeyBoardInterrupt")
      raise KeyboardInterrupt

  def close(self):
    """
    This method close properly the frame grabber.

    It releases the allocated memory and stops the acquisition.

    Returns:
        void return function.
    """
    if self.ximea and self.ximea.isOpened():
      self.ximea.release()
      print("cam closed")
    self.ximea = None

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
    return " Exposure: {0} \n Numdevice: {2} \n Width: {3} \n Height: {4} " \
           "\n X offset: {5} \n Y offset: {6}".format(self.exposure, 
                                  self.numdevice, self.width,
                                  self.height, self.xoffset, self.yoffset)
