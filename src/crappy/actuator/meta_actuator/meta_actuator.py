# coding: utf-8

from ..._global import DefinitionError


class MetaActuator(type):
  """Metaclass ensuring that two Actuators don't have the same name, and
  keeping track of all the :class:`~crappy.actuator.Actuator` classes. It also
  allows including the user-defined Actuators.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0
     not checking anymore for mandatory methods in :meth:`__init__`
  """

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
