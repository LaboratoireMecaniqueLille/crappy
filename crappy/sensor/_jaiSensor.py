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
  
  def _set_w(self,val):
    if val % 640 != 0:
      return
    sleep(.2)
    self.stopAcq()
    sleep(.2)
    CLCamera._set_w(self,val)
    sleep(.2)
    print('DEBUG new w setter',val)
    self.cap.serialWrite('WTC={}\r\n'.format(val))
    sleep(.2)
    self.startAcq()
  
  def _set_h(self,val):
    if val % 512 != 0:
      return
    sleep(.5) # DEBUG
    self.stopAcq()
    sleep(.5)
    CLCamera._set_h(self,val)
    sleep(.5)
    print('DEBUG new h setter',val)
    self.cap.serialWrite('HTL={}\r\n'.format(val))
    sleep(.5)
    self.startAcq()
    
  def get_image(self):
    print('DEBUG PY Getting image')
    t,f = CLCamera.get_image(self)
    #r = self.cap.read() #DEBUG
    print('DEBUG PY Got image')
    print('DEBUG PY', f.shape)
    print('DEBUG PY', f[25,54])
    #print('DEBUG PY', f[0,0])
    print('DEBUG PY returning...')
    return t,f
  
  def close(self):
    CLCamera.close(self)
    
  def open(self,**kwargs):
    CLCamera.open(self,**kwargs)