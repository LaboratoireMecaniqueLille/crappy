# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup StreamerCamera StreamerCamera
# @{

## @file _streamerCamera.py
# @brief Streams pictures.
# @author V1ctor Couty, Robin Siemiatkowski
# @version 0.2
# @date 10/01/2017

from _masterblock import MasterBlock
import os
import time
import sys
from crappy.technical import TechnicalCamera as tc
from ..links._link import TimeoutError
import SimpleITK as sitk

class StreamerCamera(MasterBlock):
  """
  Streams pictures.
  """

  def __init__(self,**kwargs):
    """
    This block fetch images from a camera object, can save and/or transmit them to another block.

    It can be triggered by a Link or internally
    by defining the frequency.

    Args:
        camera : {"Ximea","Jai"}
            See sensor.cameraSensor documentation.
        numdevice : int, default = 0
            If you have several camera plugged, choose the right one.
        freq : float or int or None, default=None
            Wanted acquisition frequency. Cannot exceed acquisition device capability.
            If None, will go as fast as possible.
        save : boolean, default =False
            Set to True if you want to save images.
        save_directory : directory, default = "./images/"
            directory to the saving folder. If inexistant, will be created.
        label : string, default="cycle"
            label of the input data you want to save in the name of the saved image, in
            case of external trigger.
        xoffset: int, default = 0
            Offset on the x axis.
        yoffset: int, default = 0
            Offset on the y axis.
        width: int, default = 2048
            Width of the image.
        height: int, default = 2048
            Height of the image.
    """
    MasterBlock.__init__(self)
    for arg,default in [("camera","ximea"),
                        ("numdevice",0),
                        ("freq",None),
                        ("save",False),
                        ("save_directory","./images/"),
                        ("label","cycle"),
                        ("xoffset",0),
                        ("yoffset",0),
                        ("width",2048),
                        ("height",2048),
                        ("show_fps",False)]:
      setattr(self,arg,kwargs.get(arg,default))
    self.camera_name = self.camera

    if self.save and not os.path.exists(self.save_directory):
      os.makedirs(self.save_directory)

  def prepare(self):
    self.camera = tc(self.camera_name, self.numdevice)
    for attr in ['gain','exposure','width','height']:
      setattr(self,attr,getattr(self.camera,attr))
    self.xoffset = self.camera.x_offset
    self.yoffset = self.camera.y_offset
    self.camera.sensor.new(exposure=self.exposure, width=self.width, 
                           height=self.height, xoffset=self.xoffset, 
                           yoffset=self.yoffset, gain=self.gain)
    self.trigger = "internal" if len(self.inputs) == 0 else "external"

  def main(self):
    timer = time.time()
    fps_timer = timer
    data = {}
    last_index = 0
    loops = 0
    try:
      while True:
        if self.show_fps and timer - fps_timer > 2:
          sys.stdout.write("\r[StreamerCamera] FPS: %2.2f" % (
                          (loops - last_index) / (timer - fps_timer)))
          sys.stdout.flush()
          fps_timer = timer
          last_index = loops
        if self.trigger == "internal":
          if self.freq is not None:
            while time.time() - timer < 1. / self.freq:
              pass
          timer = time.time()
          img = self.camera.sensor.get_image()
        elif self.trigger == "external":
          data = self.inputs[0].recv()  # wait for a signal
          if data is None:
            continue
          img = self.camera.sensor.get_image()
          t = time.time() - self.t0
        else:
          print "[streamCamera] What kind of trigger si that ?", self.trigger
          raise NotImplementedError
        if self.save:
          image = sitk.GetImageFromArray(img)
          try:
            cycle = data[self.label] # If we received a data to add in the name
            sitk.WriteImage(image,
                   self.save_directory + "img_%.6d_cycle%09.1f_%.5f.tiff" % (
                   loops, cycle, time.time() - self.t0))
          except KeyError: # If we did not
            sitk.WriteImage(image,
                   self.save_directory + "img_%.6d_%.5f.tiff" % (
                   loops, time.time() - self.t0))
        loops += 1
        self.send(img)
    except KeyboardInterrupt:
      self.camera.sensor.close()
      raise
    except Exception as e:
      print "Exception in streamerCamera : ", e
      self.camera.sensor.close()
      raise
