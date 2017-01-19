# coding: utf-8
##  @addtogroup technical
# @{

##  @defgroup TechnicalCamera TechnicalCamera
# @{

## @file _technicalCamera.py
# @brief Opens a camera device and initialise it (with cameraInit found in crappy[2]/technical)
#
# @author Victor Couty, Robin Siemiatkowski
# @version 0.1
# @date 17/01/2017
from __future__ import print_function


from multiprocessing import Process, Pipe
from . import get_camera_config
from . import camera_config
from crappy.sensor._meta import MetaCam




class TechnicalCamera(object):
  """
  Opens a camera device and initialise it.
  """

  def __init__(self, camera="ximea", num_device=0, videoextenso=None):
    """
    This Class opens a device and runs the initialisation sequence (CameraInit).

    It then closes the device and keep the parameters in memory for later use.
    Args:
        camera : {'ximea','jai'}, default = 'ximea'
            Name of the desired camera device.
        num_device : int, default = 0
            Number of the desired device.
        videoextenso : dict
        dict of parameters that you can use to pass informations.

        * 'enabled' : Bool
            Set True if you need the videoextenso.
        * 'white_spot' : Bool
            Set to True if your spots are white on a dark material.
        * 'border' : int, default = 4
            Size of the border for spot detection
        * 'x_offset' : int
            Offset for the x-axis.
        * 'y_offset' : int
            Offset for the y-axis
        * 'height' : int
            Height of the image, in pixels.
        * 'width : int
            Width of the image, in pixels.
    """
    try:
      camera_class = MetaCam.classes[camera]
    except KeyError:
      print("Could not find camera",camera,
                    "\nAvailables cameras are:",MetaCam.classes.keys())
      raise NotImplementedError("Could not find camera "+camera)
    
    # ======= Serial stuff: for jai cam ? ======
    try:
      module = __import__("crappy.sensor.clserial", fromlist=[camera.capitalize() + "Serial"])
      code_class = getattr(module, camera.capitalize() + "Serial")
      from crappy.sensor.clserial import ClSerial as cl
      ser = code_class()
      self.serial = cl(ser)
    except ImportError:
      self.serial = None
    except Exception as e:
      self.serial = None
    # print "module, camera_class, serial : ", module, camera_class, self.serial
    # ========================

    # initialisation:
    self.sensor = camera_class(numdevice=num_device)
    #data = get_camera_config(self.sensor)
    self.sensor.open()
    camera_config(self.sensor)
    

  def __getattr__(self,attr):
    try:
      return super(TechnicalCamera,self).__getattr__(attr)
    except AttributeError:
      try:
        return getattr(self.sensor,attr)
      except RuntimeError: # To avoid recursion error if self.sensor is not set
        raise AttributeError("TechnicalCamera has no attribute "+attr)

  def set(self,setting,value):
    setattr(self.sensor,setting,value)

  def __str__(self):
    return self.sensor.__str__()
