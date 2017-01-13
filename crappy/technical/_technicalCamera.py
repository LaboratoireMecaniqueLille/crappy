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
# @date 13/07/2016

from multiprocessing import Process, Pipe
from . import get_camera_config


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
      module = __import__("crappy.sensor", fromlist=[camera.capitalize()])
      camera_class = getattr(module, camera.capitalize())
    except AttributeError as e:
      print "{0}".format(e), " : Unreconized camera\n"
      raise
    try:
      module = __import__("crappy.sensor.clserial", fromlist=[camera.capitalize() + "Serial"])
      code_class = getattr(module, camera.capitalize() + "Serial")
      from crappy.sensor.clserial import ClSerial as cl
      ser = code_class()
      self.serial = cl(ser)
    except ImportError:
      self.serial = None
    except Exception as e:
      print "{0}".format(e)
      self.serial = None
    # print "module, camera_class, serial : ", module, camera_class, self.serial
    # initialisation:
    self.sensor = camera_class(numdevice=num_device)
    self.video_extenso = videoextenso
    data = get_camera_config(self.sensor,self.video_extenso)
    if self.video_extenso and self.video_extenso['enabled']:
      self.exposure, self.gain, self.width, self.height, self.x_offset, self.y_offset, self.minx, self.max_x, \
      self.miny, self.maxy, self.NumOfReg, self.L0x, self.L0y, self.thresh, self.Points_coordinates = data[:]
    else:
      self.exposure, self.gain, self.width, self.height, self.x_offset, self.y_offset = data[:]

  def __str__(self):
    return self.sensor.__str__()
