# coding: utf-8

from .._global import DefinitionError


class MetaModifier(type):
  """To keep track of all Modifiers (formerly Conditions)"""

  classes = {}
  needed_methods = ["evaluate"]

  def __new__(mcs, name, bases, dict_):
    return type.__new__(mcs, name, bases, dict_)

  def __init__(cls, name, bases, dict_):
    type.__init__(cls, name, bases, dict_)
    if name in MetaModifier.classes:
      raise DefinitionError("Cannot redefine " + name + " class")

    missing_methods = []
    for m in MetaModifier.needed_methods:
      if m not in dict_:
        missing_methods.append(m)
    if name != "Modifier" and missing_methods:
      raise DefinitionError("Class " + name + " is missing methods: " + str(
        missing_methods))
    del missing_methods
    MetaModifier.classes[name] = cls


class Modifier(object):
  __metaclass__ = MetaModifier
