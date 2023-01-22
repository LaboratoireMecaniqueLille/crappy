# coding: utf-8

from ..._global import DefinitionError


class MetaActuator(type):
  """Metaclass ensuring that two Actuators don't have the same name, and that
  all Actuators define the required methods. Also keeps track of all the
  Actuator classes, including the custom user-defined ones."""

  classes = {}

  def __new__(mcs, name: str, bases: tuple, dct: dict) -> type:
    return super().__new__(mcs, name, bases, dct)

  def __init__(cls, name: str, bases: tuple, dct: dict) -> None:
    super().__init__(name, bases, dct)

    # Checking that an InOut with the same name doesn't already exist
    if name in cls.classes:
      raise DefinitionError(f"The {name} class is already defined !")

    # Otherwise, saving the class
    cls.classes[name] = cls
