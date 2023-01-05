# coding: utf-8

from typing import Optional
import logging
from multiprocessing import current_process

from .._global import DefinitionError


class MetaModifier(type):
  """Metaclass keeping track of all the Modifiers, including the custom
  user-defined ones."""

  classes = {}

  needed_methods = ["evaluate"]

  def __new__(mcs, name: str, bases: tuple, dct: dict) -> type:
    return super().__new__(mcs, name, bases, dct)

  def __init__(cls, name: str, bases: tuple, dct: dict) -> None:
    super().__init__(name, bases, dct)

    # Checking that a Modifier with the same name doesn't already exist
    if name in cls.classes:
      raise DefinitionError(f"The {name} class is already defined !")

    # Gathering all the defined methods
    defined_methods = list(dct.keys())
    defined_methods += [base.__dict__.keys() for base in bases]

    # Checking for missing methods
    missing_methods = [meth for meth in cls.needed_methods
                       if meth not in defined_methods]

    # Raising if there are unexpected missing methods
    if missing_methods and name != "Modifier":
      raise DefinitionError(
        f'Class {name} is missing the required method(s): '
        f'{", ".join(missing_methods)}')

    # Otherwise, saving the class
    if name != "Modifier":
      cls.classes[name] = cls


class Modifier(metaclass=MetaModifier):
  """The base class for all modifier classes, simply allowing to keep track of
  them."""

  def __init__(self) -> None:
    """"""

    self._logger: Optional[logging.Logger] = None

  def log(self, level: int, msg: str) -> None:
    """"""

    if self._logger is None:
      self._logger = logging.getLogger(
        f"crappy.{current_process().name}.{type(self).__name__}")

    self._logger.log(level, msg)
