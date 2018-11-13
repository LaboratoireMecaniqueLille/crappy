# coding: utf-8


from time import sleep,time

from .camera import Camera
from . import ximeaModule as xi

xi_format_dict = {'8 bits': 0, '10 bits': 1, '8 bits RAW': 5, '10 bits RAW': 6}


class Ximea(Camera):
  """
  Camera class for ximea devices, this class should inherit from Camera
  This class cannot go beyond 30~35 fps depending on your hardware, but does
  not require openCV3 with Ximea flags
  """

  def __init__(self):
    """
    Args:
        numdevice: Device number
    """
    Camera.__init__(self)
    self.name = "Ximea"
    self.ximea = None
    self.add_setting("width",self._get_w,self._set_w,(1,self._get_w))
    self.add_setting("height",self._get_h,self._set_h,(1,self._get_h))
    self.add_setting("xoffset",self._get_ox,self._set_ox,(0,self._get_w))
    self.add_setting("yoffset",self._get_oy,self._set_oy,(0,self._get_h))
    self.add_setting("exposure",self._get_exp,self._set_exp,(28,100000),10000)
    self.add_setting("gain",self._get_gain,self._set_gain,(0.,6.))
    self.add_setting("data_format",self._get_data_format,
                                   self._set_data_format,xi_format_dict)
    self.add_setting("AEAG",self._get_AEAG,self._set_AEAG,True,False)
    self.add_setting("External Trigger",setter=self._set_ext_trig,limits=True,
                                                          default=False)

  def open(self,numdevice=0, **kwargs):
    """
    Will actually open the camera, args will be set to default unless
    specified otherwise in kwargs
    """
    self.numdevice = numdevice
    self.close()# If it was already open (won't do anything if cam is not open)

    if type(self.numdevice) == str:
      # open the ximea device Ximea devices start at 1100. 1100
      # => device 0, 1101 => device 1
      self.ximea = xi.VideoCapture(device_path=self.numdevice)
    else:
      self.ximea = xi.VideoCapture(self.numdevice)
    # Will apply all the settings to default or specified value
    self.set_all(**kwargs)
    self.set_all(**kwargs)

  def reopen(self, **kwargs):
    """
    Will reopen the camera
    """
    self.close()# If it was already open (won't do anything if cam is not open)

    if type(self.numdevice) == str:
      # open the ximea device Ximea devices start at 1100. 1100
      # => device 0, 1101 => device 1
      self.ximea = xi.VideoCapture(device_path=self.numdevice)
    else:
      self.ximea = xi.VideoCapture(self.numdevice)
    # Will apply all the settings to default or specified value
    self.set_all(override=True,**kwargs)

  def _get_w(self):
    return self.ximea.get(xi.CAP_PROP_FRAME_WIDTH)

  def _get_h(self):
    return self.ximea.get(xi.CAP_PROP_FRAME_HEIGHT)

  def _get_ox(self):
    return self.ximea.get(xi.CAP_PROP_XI_OFFSET_X)

  def _get_oy(self):
    return self.ximea.get(xi.CAP_PROP_XI_OFFSET_Y)

  def _get_gain(self):
    return self.ximea.get(xi.CAP_PROP_GAIN)

  def _get_exp(self):
    return self.ximea.get(xi.CAP_PROP_EXPOSURE)

  def _get_data_format(self):
    return self.ximea.get(xi.CAP_PROP_XI_DATA_FORMAT)

  def _get_AEAG(self):
    return self.ximea.get(xi.CAP_PROP_XI_AEAG)

  def _set_w(self,i):
    self.ximea.set(xi.CAP_PROP_FRAME_WIDTH,i)

  def _set_h(self,i):
    self.ximea.set(xi.CAP_PROP_FRAME_HEIGHT,i)

  def _set_ox(self,i):
    self.ximea.set(xi.CAP_PROP_XI_OFFSET_X,i)

  def _set_oy(self,i):
    self.ximea.set(xi.CAP_PROP_XI_OFFSET_Y,i)

  def _set_gain(self,i):
    self.ximea.set(xi.CAP_PROP_GAIN,i)

  def _set_exp(self,i):
    self.ximea.set(xi.CAP_PROP_EXPOSURE,i)

  def _set_ext_trig(self,i):
    self.ximea.addTrigger(1000000, int(bool(i)))

  def _set_data_format(self,i):
    # 0=8 bits, 1=16(10)bits, 5=8bits RAW, 6=16(10)bits RAW
    if i == 1 or i == 6:
      # increase the FPS in 10 bits
      self.ximea.set(xi.CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH, 10)
      self.ximea.set(xi.CAP_PROP_XI_DATA_PACKING, 1)
    self.ximea.set(xi.CAP_PROP_XI_DATA_FORMAT, i)

  def _set_AEAG(self,i):
    return self.ximea.set(xi.CAP_PROP_XI_AEAG,int(bool(i)))

  def get_image(self):
    """
    This method get a frame on the selected camera and return a ndarray

    If the camera breaks down, it reinitializes it, and tries again.

    Returns:
        frame from ximea device (ndarray height*width)
    """
    ret, frame = self.ximea.read()
    t = time()
    while 0 in frame['data'].shape or frame['data'].size >= 100000000:
      ret, frame = self.ximea.read()
      t = time()
    try:
      if ret:
        return t,frame.get('data')
      print("restarting camera...")
      sleep(2)
      self.reopen(**self.settings_dict)
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
    return " Exposure: {0} \n Numdevice: {1} \n Width: {2} \n Height: {3} " \
           "\n X offset: {4} \n Y offset: {5}".format(self.exposure,
                                  self.numdevice, self.width,
                                  self.height, self.xoffset, self.yoffset)
