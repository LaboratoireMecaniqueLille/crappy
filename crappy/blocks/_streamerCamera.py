# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup StreamerCamera StreamerCamera
# @{

## @file _streamerCamera.py
# @brief Streams pictures.
# @author Robin Siemiatkowski
# @version 0.1
# @date 13/07/2016

from _masterblock import MasterBlock
import os
import time
from crappy.technical import TechnicalCamera as tc
from ..links._link import TimeoutError
import SimpleITK as sitk

class StreamerCamera(MasterBlock):
  """
  Streams pictures.
  """

  def __init__(self, camera, numdevice=0, freq=None, save=False, save_directory="./images/", label="cycle", xoffset=0,
               yoffset=0, width=2048, height=2048, show_fps=False):
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
    self.camera_name = camera
    self.numdevice = numdevice
    camera = tc(camera, self.numdevice,
                     videoextenso={'enabled': False, 'xoffset': xoffset, 'yoffset': yoffset, 'width': width,
                                   'height': height})
    self.freq = freq
    self.save = save
    self.i = 0
    self.save_directory = save_directory
    self.label = label
    self.width = camera.width
    self.height = camera.height
    self.xoffset = camera.x_offset
    self.yoffset = camera.y_offset
    self.exposure = camera.exposure
    self.gain = camera.gain
    self.show_fps = show_fps
    if not os.path.exists(self.save_directory) and self.save:
      os.makedirs(self.save_directory)

  def main(self):
    self.camera = tc(self.camera_name, self.numdevice,
                     videoextenso={'enabled': False, 'xoffset': self.xoffset, 
                     'yoffset': self.yoffset, 'width': self.width,
                                   'height': self.height},config=False)
    self.camera.sensor.new(self.exposure, self.width, self.height, self.xoffset, self.yoffset, self.gain)
    trigger = "internal" if len(self.inputs) == 0 else "external"
    timer = time.time()
    fps_timer = timer
    loops = 0
    try:
      while True:
        loops += 1
        if self.show_fps and timer - fps_timer > 2:
          print "[StreamerCamera] FPS:", loops / (timer - fps_timer)
          fps_timer = timer
          loops = 0
        if trigger == "internal":
          if self.freq is not None:
            while time.time() - timer < 1. / self.freq:
              pass
          timer = time.time()
          try:
            img = self.camera.sensor.get_image()
          except Exception as e:
            print e
            raise
          if self.save:
            image = sitk.GetImageFromArray(img)
            sitk.WriteImage(image,
                                 self.save_directory + "img_%.6d_%.5f.tiff" % (self.i, time.time() - self.t0))
            self.i += 1
        elif trigger == "external":
          Data = self.inputs[0].recv()  # wait for a signal
          if Data is not None:
            img = self.camera.sensor.get_image()
            t = time.time() - self.t0
            if self.save:
              image = sitk.GetImageFromArray(img)
              try:
                sitk.WriteImage(image,
                                     self.save_directory + "img_%.6d_cycle%09.1f_%.5f.tiff" % (
                                       self.i, Data[self.label], time.time() - self.t0))
              except KeyError:
                sitk.WriteImage(image,
                                     self.save_directory + "img_%.6d_%.5f.tiff" % (self.i, time.time() - self.t0))
              self.i += 1
        try:
          if trigger == "internal" or Data is not None:
            for output in self.outputs:
              output.send(img)
        except TimeoutError:
          raise
        except AttributeError:  # if no outputs
          pass
    except KeyboardInterrupt:
      self.camera.sensor.close()
      # raise
    except Exception as e:
      print "Exception in streamerCamera : ", e
      self.camera.sensor.close()
      # raise
