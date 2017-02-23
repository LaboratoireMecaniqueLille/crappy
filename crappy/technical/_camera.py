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
# @date 10/02/2017
from __future__ import print_function


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
        kwargs starting with "INIT_" will be given to the sensor when instanciating
        ("INIT_" will be removed)
        the others will be transfered as is when opening it.
    """
    try:
      camera_class = MetaCam.classes[camera]
    except KeyError:
      print("Could not find camera",camera,
                    "\nAvailables cameras are:",MetaCam.classes.keys())
      raise NotImplementedError("Could not find camera "+camera)

    self.init_kwargs = {}
    self.open_kwargs = {}
    for arg in kwargs:
      if arg[:5] == 'INIT_':
        self.init_kwargs[arg[5:]] = kwargs[arg]
      else:
        self.open_kwargs[arg] = kwargs[arg]

    # initialisation:
    self.sensor = camera_class(numdevice=num_device,**self.init_kwargs)
    self.sensor.open(**self.open_kwargs)
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
