#coding: utf-8

from .._global import DefinitionError

class MetaCondition(type):
  """
  To keep track of all conditions
  """
  classes = {}
  needed_methods = ["evaluate"]

  def __new__(metacls,name,bases,dict):
    return type.__new__(metacls, name, bases, dict)

  def __init__(cls,name,bases,dict):
    type.__init__(cls,name,bases,dict)
    if name in MetaCondition.classes:
      raise DefinitionError("Cannot redefine "+name+" class")

    missing_methods = []
    for m in MetaCondition.needed_methods:
      if not m in dict:
        missing_methods.append(m)
    if name != "Condition" and missing_methods:
      raise DefinitionError("Class "+name+" is missing methods: "+str(
                                                      missing_methods))
    del missing_methods
    MetaCondition.classes[name] = cls

class Condition(object):
  __metaclass__ = MetaCondition
