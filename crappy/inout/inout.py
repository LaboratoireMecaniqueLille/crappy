# coding: utf-8

from time import time

from .._global import DefinitionError


class MetaIO(type):
  """ Metaclass that will define all IO objects.

  All IO classes should be of this type.

  To do so, simply add ``__metaclass__ = MetaIO`` in the class definition.
  (Obviously, you must import this Metaclass first.)

  MetaIO is a MetaClass: we will NEVER do ``c = MetaIO(...)``.

  The :meth:`__init__` is used to init the classes of type MetaIO (with
  ``__metaclass__ = MetaIO`` as a class attribute) and NOT an instance of
  MetaClass.
  """

  classes = {}  # This dict will keep track of all the existing cam classes
  # Attention: It keeps track of the CLASSES, not the instances !
  # If a class is defined without these
  IOclasses = {}  # Classes that are inputs and outputs
  Oclasses = {}  # Classes that only outputs
  Iclasses = {}  # Classes that only inputs
  needed_methods = ["open", "close"]

  # methods, it will raise an error

  def __new__(mcs, name, bases, dict_):
    # print "[MetaIO.__new__] Creating class",name,"from metaclass",mcs
    return type.__new__(mcs, name, bases, dict_)

  def __init__(cls, name, bases, dict_):
    # print "[MetaIO.__init__] Initializing",cls
    type.__init__(cls, name, bases, dict_)  # This is the important line
    # It creates the class, the same way we could do this:
    # MyClass = type(name,bases,dict_)
    # bases is a tuple containing the parents of the class
    # dict is the dict with the methods

    # MyClass = type("MyClass",(object,),{'method': do_stuff})
    # is equivalent to
    # class MyClass(object):
    #   def method():
    #       do_stuff()

    # Check if this class hasn't already been created
    if name in MetaIO.classes:
      raise DefinitionError("Cannot redefine " + name + " class")
    # Check if mandatory methods are defined
    missing_methods = []
    for m in MetaIO.needed_methods:
      if m not in dict_:
        missing_methods.append(m)
    if name != "InOut":
      if missing_methods:
        raise DefinitionError("Class " + name + " is missing methods: " + str(
          missing_methods))
      i = ("get_data" in dict_ or "get_stream" in dict_)
      o = ("set_cmd" in dict_)
      if i and o:
        MetaIO.IOclasses[name] = cls
      elif i:
        MetaIO.Iclasses[name] = cls
      elif o:
        MetaIO.Oclasses[name] = cls
      else:
        raise DefinitionError(
          name + " needs at least get_data, get_stream or set_cmd method")
      MetaIO.classes[name] = cls


class InOut(object, metaclass=MetaIO):
  @classmethod
  def is_input(cls):
    return hasattr(cls, 'get_data')

  @classmethod
  def is_output(cls):
    return hasattr(cls, 'set_cmd')

  def eval_offset(self):
    assert self.is_input(), "eval_offset only works for inputs!"
    if not hasattr(self, 'eval_offset_delay'):
      self.eval_offset_delay = 2  # Default value
    t0 = time()
    table = []
    while True:
      if time() > t0 + self.eval_offset_delay:
        break
      table.append(self.get_data()[1:])
    ret = []
    for i in zip(*table):
      ret.append(-sum(i) / len(i))
    return ret
