# coding: utf-8
##  @addtogroup technical
# @{

##  @defgroup TechnicalCamera TechnicalCamera
# @{

## @file _camera.py
# @brief Opens a camera device and initialises it
#
# @author Victor Couty, Robin Siemiatkowski
# @version 0.2
# @date 26/01/2017
from __future__ import print_function


from . import get_camera_config
from . import camera_config
from crappy.sensor._meta import MetaCam


class TechnicalCamera(object):
  """
  Opens a camera device and initialise it.
  """

  def __init__(self, camera="Ximea", num_device=0, config=True, **kwargs):
    """
    This Class opens a device and runs the initialisation sequence (CameraInit).

    It then closes the device and keep the parameters in memory for later use.
    Args:
        camera : {'Ximea','Webcam',...}, default = 'Ximea'
            Name of the desired camera device.
            All availalbe cameras are in crappy.sensor._meta.MetaCam.classes
        num_device : int, default = 0
            Number of the desired device.
            Width of the image, in pixels.
        config: Call the configurator ? Bool, default= True
        kwargs will be transmitted to the camera sensor
    """
    try:
      camera_class = MetaCam.classes[camera]
    except KeyError:
      print("Could not find camera",camera,
                    "\nAvailables cameras are:",MetaCam.classes.keys())
      raise NotImplementedError("Could not find camera "+camera)
    
    # initialisation:
    self.sensor = camera_class(numdevice=num_device)
    self.sensor.open(**kwargs)
    if config:
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
