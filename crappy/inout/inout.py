#coding: utf-8

from __future__ import print_function,division

#from .._global import DefinitionError
class DefinitionError(Exception):
  pass

class MetaIO(type):
  """
  Metaclass that will define all IO objects
  All IO classes should be of this type
  To do so, simply add __metaclass__ = MetaIO in the class definition
  (Obviously, you must import this Metaclass first)
  """
  classes = {} #This dict will keep track of all the existing cam classes
  #Attention: It keeps track of the CLASSES, not the instances !
  #If a class is defined without these
  IOclasses = {} #Classes that are inputs and outputs
  Oclasses = {} #Classes that only outputs
  Iclasses = {} #Classes that only inputs
  needed_methods = ["open","close"]
  #methods, it will raise an error

  def __new__(metacls,name,bases,dict):
    #print "[MetaIO.__new__] Creating class",name,"from metaclass",metacls
    return type.__new__(metacls, name, bases, dict)

  def __init__(cls,name,bases,dict):
    """
    Note: MetaIO is a MetaClass: we will NEVER do c = MetaIO(...)
    This __init__ is used to init the classes of type MetaIO
    (with __metaclass__ = MetaIO as a class attribute)
    and NOT an instance of MetaClass
    """
    #print "[MetaIO.__init__] Initializing",cls
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
    if name in MetaIO.classes:
      raise DefinitionError("Cannot redefine "+name+" class")
    # Check if mandatory methods are defined
    missing_methods = []
    for m in MetaIO.needed_methods:
      if not m in dict:
        missing_methods.append(m)
    if name != "InOut":
      if missing_methods:
        raise DefinitionError("Class "+name+" is missing methods: "+str(
                                                      missing_methods))
      i = ("get_data" in dict)
      o = ("set_cmd" in dict)
      if i and o:
        MetaIO.IOclasses[name] = cls
      elif i:
        MetaIO.Iclasses[name] = cls
      elif o:
        MetaIO.Oclasses[name] = cls
      else:
        raise DefinitionError(
              name+" needs at least get_data or set_cmd method")
      MetaIO.classes[name] = cls

class InOut(object):
  __metaclass__ = MetaIO

  @classmethod
  def is_input(cls):
    return hasattr(cls,'get_data')

  @classmethod
  def is_output(cls):
    return hasattr(cls,'set_cmd')
