#coding: utf-8
from __future__ import print_function
from time import sleep

from ._clSensor import CLCamera

class Jai(CLCamera):
  def __init__(self, **kwargs):
    kwargs['camera_type'] = "FullAreaGray8"
    CLCamera.__init__(self,**kwargs)
    self.settings['width'].limits = (1,2560)
    self.settings['width'].default = 2560
    self.settings['height'].limits = (1,2048)
    self.settings['height'].default = 2048
    self.add_setting('exposure',setter=self._set_exp,getter=self._get_exp,
                               limits = (10,800000))

  def _set_w(self,val):
    if val % 640 != 0:
      return
    self.stopAcq()
    CLCamera._set_w(self,val)
    self.cap.serialWrite('WTC={}\r\n'.format(val))
    self.startAcq()

  def _set_h(self,val):
    if val % 512 != 0:
      return
    self.stopAcq()
    CLCamera._set_h(self,val)
    self.cap.serialWrite('HTL={}\r\n'.format(val))
    self.startAcq()

  def _set_exp(self,val):
    self.cap.serialWrite('PE={}\r\n'.format(val))

  def _get_exp(self):
    return self.cap.serialWrite('PE?\r\n').strip()[3:]

  def get_image(self):
    return CLCamera.get_image(self)

  def close(self):
    CLCamera.close(self)

  def open(self,**kwargs):
    CLCamera.open(self,**kwargs)
