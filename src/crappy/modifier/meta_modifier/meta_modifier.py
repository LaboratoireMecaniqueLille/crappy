# coding: utf-8

from ..._global import DefinitionError


class MetaModifier(type):
  """Metaclass keeping track of all the Modifiers, including the custom
  user-defined ones.

  .. versionadded:: 1.4.0
  .. versionchanged:: 1.5.10
     not checking anymore for mandatory method in :meth:`__init__`
  """

  classes = {}

  def __new__(mcs, name: str, bases: tuple, dct: dict) -> type:
    return super().__new__(mcs, name, bases, dct)

  def __init__(cls, name: str, bases: tuple, dct: dict) -> None:
    super().__init__(name, bases, dct)

    if hasattr(cls, 'evaluate'):
      raise DefinitionError("The evaluate method is deprecated for Modifiers "
                            "since version 2.0.0, just rename it to __call__ "
                            "to get your Modifier working again.")

    # Checking that a Modifier with the same name doesn't already exist
    if name in cls.classes:
      raise DefinitionError(f"The {name} class is already defined !")

    # Otherwise, saving the class
    cls.classes[name] = cls
