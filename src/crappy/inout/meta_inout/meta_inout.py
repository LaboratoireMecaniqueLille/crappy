# coding: utf-8

from ..._global import DefinitionError


class MetaIO(type):
  """Metaclass ensuring that two InOuts don't have the same name, and that all
  InOuts define the required methods. Also keeps track of all the InOut
  classes, including the custom user-defined ones.

  .. versionadded:: 1.4.0
  .. versionchanged:: 1.5.10
     not checking anymore for mandatory methods in :meth:`__init__`
  """

  classes = dict()

  def __new__(mcs, name: str, bases: tuple, dct: dict) -> type:
    return super().__new__(mcs, name, bases, dct)

  def __init__(cls, name: str, bases: tuple, dct: dict) -> None:
    super().__init__(name, bases, dct)

    # Checking that an InOut with the same name doesn't already exist
    if name in cls.classes:
      raise DefinitionError(f"The {name} class is already defined !")

    # Saving the class
    cls.classes[name] = cls
