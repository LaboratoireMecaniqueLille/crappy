# coding: utf-8
##  @addtogroup sensor
# @{

##  @defgroup Ximea Ximea
# @{

## @file _ximeaSensor.py
# @brief  Camera class for ximea devices, this class should inherit from CameraSensor
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 29/06/2016

from os import path

from ._meta import cameraSensor

here = path.abspath(path.dirname(__file__))
import ximeaModule as xi

import time
import platform


def resettable(f, *args, **kwargs):
  """
  Decorator for resetting the camera device. Not working yet on Windows.
  """
  import copy

  def __init_and_copy__(self, *args, **kwargs):
    if not platform.system() == "Linux":
      return f(self, **kwargs)
    f(self, **kwargs)
    self.__original_dict__ = copy.deepcopy(self.__dict__)

    def reset(o=self):
      o.__dict__ = o.__original_dict__

    self.reset = reset

  return __init_and_copy__


class Ximea(cameraSensor.CameraSensor):
  """
  Camera class for ximea devices, this class should inherit from CameraSensor

  Contains all the methods to open a device, resize the Zone Of Interest, and
  grab frames.

  Args:
      numdevice : int or string (device path), default = 0
          Number of your device.
      framespersec : int or float or None, default = None
          The wanted frequency for grabbing frame. DOESN'T WORK at the moment.
      external_trigger : bool, default = False
          Define to True if you want to trigg the acquyisition of a frame externally.
      data_format : int, default = 0
          Value must be in [0:7]. See documentation for more informations.
  """

  @resettable
  def __init__(self, numdevice=0, framespersec=None, external_trigger=False, data_format=0):
    """

    Args:
        numdevice: Device number
        framespersec: number of frame per sec
        external_trigger: Enable (True) or disable (False) the external trigger mode.
        data_format: wanted data format, please see the ximea documentation for more details.
    """
    ## number of frame per second wanted
    self.FPS = framespersec
    ## Device number
    self.numdevice = numdevice
    ## if true the external trigger is enabled.
    self.external_trigger = external_trigger
    ## wanted data format, please see the ximea documentation for more details.
    self.data_format = data_format
    ## default width of the frame
    self._defaultWidth = 2048
    ## default height of the frame
    self._defaultHeight = 2048
    ## default offset x
    self._defaultXoffset = 0
    ## default offset y
    self._defaultYoffset = 0
    ## default exposure time (in ms)
    self._defaultExposure = 10000
    ## default gain
    self._defaultGain = 0
    ## number of acquired frames.
    self.nbi = 0

  def new(self, exposure=10000, width=2048, height=2048, xoffset=0, yoffset=0, gain=0):
    """
    This method opens the ximea device and return a camera object.

    Args:
        exposure: exposure time in ms
        width: frame width
        height: frame height
        xoffset: frame offset in x
        yoffset: frame offset in y
        gain: gain
    Note:
        For ximea device width+xoffset must be always less or equal to 2048
        and height+yoffset must be always less or equal to 2048
    Returns:
        Ximea instance.

    """
    if platform.system() == "Linux":
      nd, fps, et, df = self.numdevice, self.FPS, self.external_trigger, self.data_format
      self.reset()
      self.__init__(numdevice=nd, framespersec=fps, external_trigger=et, data_format=df)

    GLOBAL_ENABLE_FLAG = True

    if type(self.numdevice) == str:
      # open the ximea device Ximea devices start at 1100. 1100 => device 0, 1101 => device 1
      self.ximea = xi.VideoCapture(device_path=self.numdevice)
    else:
      self.ximea = xi.VideoCapture(self.numdevice)
    if self.external_trigger:  # this condition activate the trigger mode
      self.ximea.addTrigger(1000000, True)
    self.ximea.set(xi.CAP_PROP_XI_DATA_FORMAT,
                   self.data_format)  # 0=8 bits, 1=16(10)bits, 5=8bits RAW, 6=16(10)bits RAW

    if self.data_format == 1 or self.data_format == 6:  # increase the FPS in 10 bits
      self.ximea.set(xi.CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH, 10)
      self.ximea.set(xi.CAP_PROP_XI_DATA_PACKING, 1)

    self.ximea.set(xi.CAP_PROP_XI_AEAG, 0)  # auto gain auto exposure
    ## frame width
    self.width = width
    ## frame height
    self.height = height
    ## frame x offset
    self.xoffset = xoffset
    ## frame y offset
    self.yoffset = yoffset
    ## exposure time in ms
    self.exposure = exposure
    ## gain
    self.gain = gain

  def get_image(self):
    """
    This method get a frame on the selected camera and return a ndarray

    If the camera breaks down, it reinitializes it, and tries again.

    Returns:
        grabber frame from ximea device (ndarray height*width)

    """
    self.nbi += 1
    try:
      # print self.nbi
      ret, frame = self.ximea.read()
    except KeyboardInterrupt:
      print "KeyboardInterrupt, closing camera ..."
      self.close()
      self.quit = True
      raise KeyboardInterrupt

    try:
      if ret:
        data = frame.get('data')
        return data

      print "restarting camera..."
      time.sleep(2)
      self.new(self.exposure, self.width, self.height, self.xoffset, self.yoffset,
               self.gain)  # Reset the camera instance
      return self.get_image()
    except UnboundLocalError:  # if ret doesn't exist, because of KeyboardInterrupt
      print "ximea quitting, probably because of KeyBoardInterrupt"
      pass

  def close(self):
    """
    This method close properly the frame grabber.

    It releases the allocated memory and stops the acquisition.

    Returns:
        void return function.

    """
    print "closing camera..."
    if self.ximea.isOpened():
      self.ximea.release()
      print "cam closed"
    else:
      print "cam already closed"

  def reset_zoi(self):
    """
    Re-initialize the Zone Of Interest
    """
    self.yoffset = self._defaultYoffset
    self.xoffset = self._defaultXoffset
    self.height = self._defaultHeight
    self.width = self._defaultWidth

  def set_zoi(self, width, height, xoffset, yoffset):
    """
    Define the Zone Of Interest
    """
    self.yoffset = yoffset
    self.xoffset = xoffset
    self.width = width
    self.height = height

  @property
  def height(self):
    """
    height getter.

    Returns:
        return the height parameter.
    """
    return self._height

  @height.setter
  def height(self, height):
    """
    Height setter.

    Args:
        height: new value of th height parameter
    """
    self._height = ((int(height) / 2) * 2)
    self.ximea.set(xi.CAP_PROP_FRAME_HEIGHT, self.height)

  @property
  def width(self):
    """
    Height getter

    Returns:
        return the width parameter.

    """
    return self._width

  @width.setter
  def width(self, width):
    """
    width setter
    Args:
        width: new value of the width parameter

    """
    self._width = (int(width) - int(width) % 4)
    self.ximea.set(xi.CAP_PROP_FRAME_WIDTH, self.width)

  @property
  def yoffset(self):
    """
    yoffset getter.
    Returns:
        return the yoffset parameter

    """
    return self._yoffset

  @yoffset.setter
  def yoffset(self, yoffset):
    """
    yoffset setter.
    Args:
        yoffset: new vakue for the yoffset parameter.
    """
    y_offset = ((int(yoffset) / 2) * 2)
    self._yoffset = y_offset
    self.ximea.set(xi.CAP_PROP_XI_OFFSET_Y, y_offset)

  @property
  def xoffset(self):
    """
    xoffset getter.
    Returns:
        return the value of the xoffset parameter.
    """
    return self._xoffset

  @xoffset.setter
  def xoffset(self, xoffset):
    """
    xoffset setter.
    Args:
        xoffset: new value of the xoffset parameter
    """
    # print "xoffset setter : ", xoffset
    x_offset = (int(xoffset) - int(xoffset) % 4)
    self._xoffset = x_offset
    self.ximea.set(xi.CAP_PROP_XI_OFFSET_X, x_offset)

  @property
  def exposure(self):
    """
    exposure getter.

    Returns:
        return the value of the exposure parameter
    """
    return self._exposure

  @exposure.setter
  def exposure(self, exposure):
    """
    exposure setter.
    Args:
        exposure: new value of the exposure parameter
    """
    self.ximea.set(xi.CAP_PROP_EXPOSURE, exposure)
    self._exposure = exposure

  @property
  def gain(self):
    """
    gain getter.

    Returns:
        return the value of the gain parameter.
    """
    return self._gain

  @gain.setter
  def gain(self, gain):
    """
    gain setter
    Args:
        gain: new value for the gain parameter.
    """
    self.ximea.set(xi.CAP_PROP_GAIN, gain)
    self._gain = gain

  def __str__(self):
    """
    \__str\__ method to prints out main parameter.

    Returns:
        a formated string with the value of the main parameter.

    Example:
        camera = Ximea(numdevice=0, framespersec=99)
        camera.new(exposure=10000, width=2048, height=2048, xoffset=0, yoffset=0, gain=0)
        print camera

    these lines will print out:
         \code
         Exposure: 10000
         FPS: 99
         Numdevice: 0
         Width: 2048
         Height: 2048
         X offset: 0
         Y offset: 0
         \endcode
    """
    return " Exposure: {0} \n FPS: {1} \n Numdevice: {2} \n Width: {3} \n Height: {4} " \
           "\n X offset: {5} \n Y offset: {6}".format(self.exposure, self.FPS, self.numdevice, self.width,
                                                      self.height, self.xoffset, self.yoffset)

  @property
  def name(self):
    """
    name property getter
    Returns:
        return the name of the camera.
    """
    return "ximea"
