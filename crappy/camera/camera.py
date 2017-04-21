# coding: utf-8
from __future__ import print_function,division

from time import time,sleep

from .._global import DefinitionError

class MetaCam(type):
  """
  Metaclass that will define all cameras
  Camera classes should be of this type
  To do so, simply add __metaclass__ = MetaCam in the class definition
  (Obviously, you must import this Metaclass first)
  """
  classes = {} #This dict will keep track of all the existing cam classes
  #Attention: It keeps track of the CLASSES, not the instances !
  #If a camera is defined without these
  needed_methods = ["__init__", "get_image","open","close"]
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

    # Check if this class hasn't already been created
    if name in MetaCam.classes:
      raise DefinitionError("Cannot redefine "+name+" class")
    # Check if mandatory methods are defined
    missing_methods = []
    for m in MetaCam.needed_methods:
      if not m in dict:
        missing_methods.append(m)
    if name != "Camera" and missing_methods:
      raise DefinitionError("Class "+name+" is missing methods: "+str(
                                                      missing_methods))

    del missing_methods
    MetaCam.classes[name] = cls


class Cam_setting(object):
  """This class represents an attribute of the camera that cam be set"""
  def __init__(self,name,getter,setter,limits,default):
    """
  Arguments:
    name: the name of the setting

    default: the default value, if not specified it will be set to this value

    getter: Function to read this value from the device
      If set to None, it will assume that the setting always happened correctly

    setter: A function that will be called when setting the parameter to a new
      value. Can do nothing, it will only change its value and nothing else.

    limits: It contains the available values for this parameter
      The possible types are:
       None: Values will not be tested and the parameter will not appear in
       CameraConfig

       A tuple of 2 ints/floats: Values must be between first and second value
       CameraConfig will add a scale widget to set it. If they are integers,
       all the integers between them will be accessible, if they are floats,
       the range will be divided in 1000 in the scale widget.
       Note: If the upper value is callable (a function or method), it will
       be set to the return value of this function. It allows reading the
       max value from the device.

       A Boolean: Possible values will be True or False, CameraConfig will
       add a checkbox to edit the value.
       (It can be True or False, it doesn't matter)

       A dict: Possible values are the values of the dict, CameraConfig will
       add radio buttons showing the keys, to set it to the corresponding value
  """
    self.name = name
    self.getter = getter
    self.setter = setter
    self.limits = limits
    self.default = default
    self._value = None
    if getter is None:
      self._value = default
      self.getter = lambda *args: self._value

  @property
  def value(self):
    if self._value is None:
      self._value = self.getter()
      if type(self.limits) is tuple and callable(self.limits[1]):
          self.limits = (self.limits[0],self.limits[1]())
      if self.default is None:
        self.default = self._value
    return self._value

  # Here is the interesting part: When we set value (setting.value = x),
  # we will go throught all of this, and the new value will be the actual
  # value of the setting after the operation
  @value.setter
  def value(self,i):
    _ = self.value # Detail: to make sure we called value getter once
    #if type(self.limits) == tuple:
    #  if not self.limits[0] <= i <= self.limits[1]:
    #    print("[Cam_setting] Parameter",i,"out of range ",self.limits)
    #    return
    if type(self.limits) == dict:
      if not i in self.limits.values():
        print("[Cam_setting] Parameter",i,"not available",self.limits)
        return
    elif type(self.limits) == bool:
      i = bool(i)
    # We could actually wait to see if setter is succesful before setting the
    # value, but if setter uses self.parameter, it will still be set to its old
    # value until it returns...
    self.setter(i)
    self._value = i
    new_val = self.getter()
    if new_val != i:
      print("[Cam_setting] Could not set",self.name,"to",i,"value is",new_val)
    self._value = new_val


  def __str__(self):
    if self.limits:
      return "Setting: "+str(self.name)+", value:"+str(self._value)\
      +" Limits:"+str(self.limits)
    else:
      return "Setting: "+str(self.name)+", value:"+str(self._value)

  def __repr__(self):
    return self.__str__()


class Camera(object):
  """This class represents a camera sensor. It may have settings: they
  represent all that can be set on the camera: height, width, exposure, AEAG,
  external trigger, etc...
  Each parameter is represented by a Cam_setting object: it includes the
  default value, a function to set and get parameter, etc...
  This class makes it transparent to the user: you can access a setting by
  using myinstance.setting = stuff
  It will automatically check the validity and try to set it. (see Cam_setting)
  """
  __metaclass__ = MetaCam # The magic happens here!

  def __init__(self):
    """Don't forget to call this __init__ in the children or __getattr__ will
    fall in an infinite recursion loop looknig for settings..."""
    self.settings = {}
    self.last = time()
    self.max_fps = None
    self.name = "Camera"

  @property
  def max_fps(self):
    return self._max_fps

  @max_fps.setter
  def max_fps(self,value):
    """To compute self.delay again when fps is set"""
    self._max_fps = value
    if value:
      self.delay = 1/value
    else:
      self.delay = 0

  def add_setting(self, name, getter = None, setter = lambda *l:None,
                                            limits = None, default = None):
    """Wrapper to simply add a new setting to the camera"""
    assert name not in self.settings, "This setting already exists"
    self.settings[name] = Cam_setting(name,getter,setter,limits,default)

  @property
  def available_settings(self):
    """Returns a list of available settings"""
    return map(lambda x: x.name,self.settings.values())+["max_fps"]

  @property
  def settings_dict(self):
    """Returns settings as a dict, keys are the names of the settings and
    values are setting.value"""
    d = dict(self.settings)
    for k in d:
      d[k] = d[k]._value
    return d

  def set_all(self,override=False,**kwargs):
    """Sets all the settings based on kwargs, if not specified, the setting
    will take its default value.
    If override is True, it will not assume a setting and reset it unless
    it is already default"""
    #print('DEBUG=======set_all')
    for s in self.settings:
      if s in kwargs:
        if self.settings[s].value != kwargs[s] or override:
          #print("Setting",s,"to",kwargs[s])
          self.settings[s].value = kwargs[s]
        else:
          #print(s,'is already set to',kwargs[s])
          pass
        del kwargs[s]
      elif self.settings[s].value != self.settings[s].default:
        #print(s,'is',self.settings[s].value)
        #print("Defaulting",s,"to",self.settings[s].default)
        self.settings[s].value = self.settings[s].default
      else:
        #print(s,'is already set to',self.settings[s].default)
        pass
    for k,v in kwargs.iteritems():
      setattr(self,k,v)
    #print('DEBUG====ENDset_all')

  def reset_all(self):
    """Reset all the settings to their default values"""
    self.set_all()

  def read_image(self):
    """This method is a wrapper for get_image that will limit fps to max_fps"""
    if self.delay:
      t = time()
      wait = self.last - t + self.delay
      while wait > 0:
        t = time()
        wait = self.last - t + self.delay
        sleep(max(0,wait/10))
      self.last = t
    return self.get_image()

  def __getattr__(self,i):
    """The idea is simple: if the camera has this attribute: return it
    (default behavior) else, try to find the corresponding setting and
    return its value.
    Note that we made sure to raise an attribute error if it is neither a
    camera attribute nor a setting.
    Example:
    Camera definition contains self.add_setting("width",1280,set_w)
    then
    cam = Camera()
    cam.width will return 1280
    """
    try:
      return self.__getattribute__(i)
    except AttributeError:
      try:
        #print("Returning",i,"=",self.settings[i].value)
        return self.settings[i].value
      except KeyError:
        raise AttributeError("No such attribute: "+i)
      except RuntimeError:
        print("You have likely forgotten to call Camera.__init__(self)!")
        raise AttributeError("No such attribute:"+i)

  def __setattr__(self,attr,val):
    """Same as getattr: if it is a setting, then set its value using the
    setter in the class CamSetting, else use the default behavior
    It is important to make sure we don't try to set 'settings', it would
    recursively call getattr and enter an infinite loop, hence the condition.
    Example: cam.width = 2048 will be like cam.settings['width'].value = 2048.
    It allows for simple settings of the camera"""
    if attr != "settings" and attr in self.settings:
      self.settings[attr].value = val
    else:
      super(Camera,self).__setattr__(attr,val)

  def __str__(self):
    return self.name+" camera with {} settings".format(len(self.settings))

  def __repr__(self):
    s = self.__str__()
    for i in self.settings.values():
      s+=("\n"+str(i))
    return s
