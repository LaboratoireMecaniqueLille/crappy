#coding: utf-8
from __future__ import print_function

class DefinitionError(Exception):
    """Error to raise when classes are not defined correctly"""
    def __init__(self,msg=""):
        self.msg = msg

    def __str__(self):
        return self.msg

class MetaCam(type):
    """
    Metaclass that will define all cameras
    Camera classes should be of this type
    To do so, simply add __metaclass__ = MetaCam in the class definition
    (Obviously, you must import this Metaclass first)
    """
    classes = {} #This dict will keep track of all the existing cam classes
    #Attention: It keeps track of the CLASSES, not the instances !
    needed_methods = ["__init__", "get_image","open","close"] #If a camera is defined without these 
    #methods, it will raise an error

    def __new__(metacls,name,bases,dict):
        #print "[MetaCam.__new__] Creating class",name,"from metaclass",metacls
        return type.__new__(metacls, name, bases, dict)

    def __init__(cls,name,bases,dict):
        """
        Note: MetaCam is a MetaClass: we will NEVER do c = MetaCam(...)
        This __init__ is used to init the classes of type MetaCam
        (with __metaclass__ = MetaCam as a class attribute)
        and NOT an instance of MetaClass
        """
        #print "[MetaCam.__init__] Initializing",cls
        type.__init__(cls,name,bases,dict) # This is the important line
        #It creates the class, the same way we could do this:
        #MyClass = type(name,bases,dict)
        #bases is a tuple containing the parents of the class
        #dict is the dict with the methods

        #MyClass = type("MyClass",(object,),{'method': do_stuff})
        # is equivalent to
        #class MyClass(object):
        #   def method():
        #       do_stuff()

        if name in MetaCam.classes:
            raise DefinitionError("Cannot redefine "+name+" class")
        missing_methods = []
        for m in MetaCam.needed_methods:
            if not m in dict:
                missing_methods.append(m)
        if name != "MasterCam" and missing_methods:
            raise DefinitionError("Class "+name+" is missing methods: "+str(
                                                          missing_methods))

        del missing_methods
        MetaCam.classes[name] = cls


class Cam_setting(object):
  """This class represents an attribute of the camera that cam be set"""
  def __init__(self,name,default,set_f,limits):
    self.name = name # The name of the setting
    self.default = default # its default value
    self._value = default
    self.set_f = set_f # The function that will be called to actually
    # apply this setting, it also checks the setting for validity
    # return value must be true if the setting applied correctly,
    # false otherwise
    self.limits = limits
    # You must include the boundaries in this condition (ex: 1 <= w <=2048)
    # It can be None, it will then be ignored (useful for non number settings)
    # It must be a tuple (min,max). min type will also define the type of the
    # parameter: int of float

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self,i):
    if self.limits:
      if not self.limits[0] <= i <= self.limits[1]:
        print("[Cam_setting] Parameter",i," out of range ",self.limits)
        return
    old = self._value
    self._value = i
    if not self.set_f(i):
      print("Could not set",self.name,"to",i)
      if i == self.default:
        raise RuntimeError(
                      "Could not set default value for "+self.name+": "+str(i))
      self._value = old

  def __str__(self):
    if self.limits:
      return "Setting: "+str(self.name)+", value:"+str(self._value)\
      +" Limits:"+str(self.limits)
    else:
      return "Setting: "+str(self.name)+", value:"+str(self._value)

  def __repr__(self):
    return self.__str__()


class MasterCam(object):
  """This class represents a camera sensor. It may have settings: they
  represent all that can be set on the camera: height, width, exposure, AEAG,
  external trigger, etc...
  Each parameter is represented by a Cam_setting object: it includes the
  default value, a function to check parameter validity, etc...
  This class makes it transparent to the user: you can access a setting by
  using myinstance.setting = stuff
  It will automatically check the validity and try to set it.
  """
  __metaclass__ = MetaCam

  def __init__(self):
    """Represents a camera, all cameras should inherit from this block"""
    self.settings = {}
    self.is_open = False
    self.name = "MasterCam"

  def add_setting(self,name,default,set_f=lambda a:True,limits=(1,100000)):
    assert name not in self.settings, "This setting already exists"
    self.settings[name] = \
                  Cam_setting(name,default,set_f,limits)

  def set_all(self):
    for s in self.settings:
      self.settings[s].value = self.settings[s].value

  def reset_all(self):
    for s in self.settings:
      self.settings[s].value = self.settings[s].default

  def __getattr__(self,i):
    """The idea is simple: if the camera has this attribute: return it
    (default behavior) else, try to find the corresponding setting and
    return its value.
    Note that we made sure to raise an attribute error if it is neither a
    camera attribute nor a setting.
    """
    try:
      return super(MasterCam,self).__getattr__(i)
    except AttributeError:
      try:
        return self.settings[i].value
      except KeyError:
        raise AttributeError(self.__str__()+": No such setting: "+i)
      except RuntimeError:
        print("cam.settings is likely not defined! \
You must call MasterCam.__init__(self) first in any camera.__init__!!")
        raise AttributeError("No such attribute: settings")

  def __setattr__(self,attr,val):
    """Same as getattr: if it is a setting, then set its value using the
    setter in the class CamSetting, else use the default behavior
    It is important to make sure we don't try to set 'settings', it would
    recursively call getattr and enter an infinite loop, hence the condition."""
    if attr != "settings" and attr in self.settings:
      self.settings[attr].value = val
    else:
      super(MasterCam,self).__setattr__(attr,val)

  def __str__(self):
    return self.name+" camera with {} settings".format(len(self.settings))

  def __repr__(self):
    s = self.__str__()
    for i in self.settings.values():
      s+=("\n"+str(i))
