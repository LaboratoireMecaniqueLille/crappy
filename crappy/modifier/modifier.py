#coding: utf-8

from .._global import DefinitionError


class MetaModifier(type):
  """
  To keep track of all Modifiers (former Conditions)
  """
  classes = {}
  needed_methods = ["evaluate"]

  def __new__(metacls,name,bases,dict):
    return type.__new__(metacls, name, bases, dict)

  def __init__(cls,name,bases,dict):
    type.__init__(cls,name,bases,dict)
    if name in MetaModifier.classes:
      raise DefinitionError("Cannot redefine "+name+" class")

    missing_methods = []
    for m in MetaModifier.needed_methods:
      if m not in dict:
        missing_methods.append(m)
    if name != "Modifier" and missing_methods:
      raise DefinitionError("Class "+name+" is missing methods: "+str(
        missing_methods))
    del missing_methods
    MetaModifier.classes[name] = cls


class Modifier(object):
  __metaclass__ = MetaModifier
