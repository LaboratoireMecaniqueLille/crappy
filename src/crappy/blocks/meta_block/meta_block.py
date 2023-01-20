# coding: utf-8

from ..._global import DefinitionError


class MetaBlock(type):
  """"""

  existing = list()

  def __new__(mcs, name: str, bases: tuple, dct: dict) -> type:
    return super().__new__(mcs, name, bases, dct)

  def __init__(cls, name: str, bases: tuple, dct: dict) -> None:
    super().__init__(name, bases, dct)

    # Checking that a Block with the same name doesn't already exist
    if name in cls.existing:
      raise DefinitionError(f"The {name} Block is already defined !")

    # Saving the name
    cls.existing.append(name)
