# coding: utf-8

from ..._global import DefinitionError


class MetaCamera(type):
  """Metaclass ensuring that two cameras don't have the same name, and that all
  cameras define the required methods. Also keeps track of all the Camera
  classes, including the custom user-defined ones.

  .. versionadded:: 1.4.0
  .. versionchanged:: 1.5.10
     not checking anymore for mandatory methods in :meth:`__init__`
  """

  classes = {}

  def __new__(mcs, name: str, bases: tuple, dct: dict) -> type:
    return super().__new__(mcs, name, bases, dct)

  def __init__(cls, name: str, bases: tuple, dct: dict) -> None:
    super().__init__(name, bases, dct)

    # Checking that a Camera with the same name doesn't already exist
    if name in cls.classes:
      raise DefinitionError(f"The {name} class is already defined !")

    # Otherwise, saving the class
    cls.classes[name] = cls
