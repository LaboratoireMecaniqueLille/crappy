# coding: utf-8

from ...._global import DefinitionError


class MetaPath(type):
  """Metaclass ensuring that two Paths don't have the same name, and that all
  Paths define the required methods. Also keeps track of all the Path
  classes, including the custom user-defined ones.
  
  .. versionadded:: 2.0.0
  """

  classes = dict()

  def __new__(mcs, name: str, bases: tuple, dct: dict) -> type:
    return super().__new__(mcs, name, bases, dct)

  def __init__(cls, name: str, bases: tuple, dct: dict) -> None:
    super().__init__(name, bases, dct)

    # Checking that a Path with the same name doesn't already exist
    if name in cls.classes:
      raise DefinitionError(f"The {name} Generator Path is already defined !")

    # Saving the name
    cls.classes[name] = cls
