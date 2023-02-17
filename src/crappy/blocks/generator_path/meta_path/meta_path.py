# coding: utf-8

from ...._global import DefinitionError


class MetaPath(type):
  """"""

  classes = list()

  def __new__(mcs, name: str, bases: tuple, dct: dict) -> type:
    return super().__new__(mcs, name, bases, dct)

  def __init__(cls, name: str, bases: tuple, dct: dict) -> None:
    super().__init__(name, bases, dct)

    # Checking that a Path with the same name doesn't already exist
    if name in cls.classes:
      raise DefinitionError(f"The {name} Generator Path is already defined !")

    # Saving the name
    cls.classes.append(name)
