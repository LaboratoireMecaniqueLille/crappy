# coding: utf-8

from ..._global import DefinitionError


class MetaBlock(type):
  """Metaclass ensuring that two Blocks don't have the same name, and that all
  Blocks define the required methods. Also keeps track of all the Block
  classes, including the custom user-defined ones.

  .. versionadded:: 2.0.0
  """

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
