#coding: utf-8
from __future__ import print_function

from .camera import MetaCam

class Camera_wrapper(object):
  def __init__(self,name,**kwargs):
    if not name in MetaCam.classes:
      m = "Unknown type of camera:"+name
      m += "\nAvailable cameras are:"+str(MetaCam.classes.keys())
      raise AttributeError(m)
    self._cam = MetaCam.classes[name](**kwargs)

  def __getattr__(self,a):
    try:
      return self.__getattribute__(a)
    except AttributeError:
      return getattr(self._cam,a)

