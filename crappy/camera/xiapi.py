# coding: utf-8

import time
from ximea import xiapi

from .camera import Camera


class Xiapi(Camera):
  """
  Camera class for ximeas using official XiAPI
  """
  def __init__(self):
    Camera.__init__(self)
    self.name = "Xiapi"
    self.cam = xiapi.Camera()
    self.img = xiapi.Image()
    self.add_setting("width",self._get_w,self._set_w,(1,self._get_w))
    self.add_setting("height",self._get_h,self._set_h,(1,self._get_h))
    self.add_setting("xoffset",self._get_ox,self._set_ox,(0,self._get_w))
    self.add_setting("yoffset",self._get_oy,self._set_oy,(0,self._get_h))
    self.add_setting("exposure",self._get_exp,self._set_exp,(28,100000),10000)
    self.add_setting("gain",self._get_gain,self._set_gain,(0.,6.))
    self.add_setting("data_format",self._get_data_format,
                                   self._set_data_format,xi_format_dict)
    self.add_setting("AEAG",self._get_AEAG,self._set_AEAG,True,False)
    self.add_setting("external_trig",self._get_extt,self._set_extt,True,False)

  def _get_w(self):
    pass

  def _get_h(self):
    pass

  def _get_ox(self):
    pass

  def _get_oy(self):
    pass

  def _get_gain(self):
    pass

  def _get_exp(self):
    pass

  def _get_AEAG(self):
    pass

  def _get_data_format(self):
    pass

  def _get_extt(self):
    pass

  def _set_w(self,i):
    pass

  def _set_h(self,i):
    pass

  def _set_ox(self,i):
    pass

  def _set_oy(self,i):
    pass

  def _set_gain(self,i):
    pass

  def _set_exp(self,i):
    pass

  def _set_AEAG(self,i):
    pass

  def _set_data_format(self,i):
    pass

  def _set_extt(self,i):
    pass

  def open(self,sn=None,**kwargs):
    """
    Will actually open the camera, args will be set to default unless
    specified otherwise in kwargs

    If sn is given, it will open the camera with
    the corresponing serial number

    Else, it will open any camera
    """
    self.sn = sn
    self.close()
    if self.sn is not None:
      self.cam.open_device_by_sn(self.sn)
    else:
      self.cam.open_device()

    for k in kwargs:
      assert k in self.available_settings,str(self)+"Unexpected kwarg: "+str(k)
    self.set_all(**kwargs)
    self.set_all(**kwargs)
    self.cam.start_acquisition()

  def reopen(self,**kwargs):
    """
    Will reopen the camera, args will be set to default unless
    specified otherwise in kwargs
    """
    self.open()
    self.set_all(override=True,**kwargs)

  def get_image(self):
    """
    This method get a frame on the selected camera and return a ndarray

    Returns:
        frame from ximea device (ndarray height*width)
    """
    self.cam.get_image(self.img)
    t = time.time()
    return t,frame.get_image_data_numpy()

  def close(self):
    """
    This method closes properly the camera

    Returns:
        void return function.
    """
    self.cam.close_device()
