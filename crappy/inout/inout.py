# coding: utf-8

from time import time, sleep
from typing import Optional, Dict, Any, Union

from .._global import DefinitionError


class MetaIO(type):
  """Metaclass ensuring that two InOuts don't have the same name, and that all
  InOuts define the required methods. Also keeps track of all the InOut
  classes, including the custom user-defined ones."""

  classes = {}
  needed_methods = ["open", "close"]

  def __new__(mcs, name: str, bases: tuple, dct: dict) -> type:
    return super().__new__(mcs, name, bases, dct)

  def __init__(cls, name: str, bases: tuple, dct: dict) -> None:
    super().__init__(name, bases, dct)

    # Checking that an InOut with the same name doesn't already exist
    if name in cls.classes:
      raise DefinitionError(f"The {name} class is already defined !")

    # Gathering all the defined methods
    defined_methods = list(dct.keys())
    defined_methods += [base.__dict__.keys() for base in bases]

    # Checking for missing methods
    missing_methods = [meth for meth in cls.needed_methods
                       if meth not in defined_methods]

    # Raising if there are unexpected missing methods
    if missing_methods and name != "InOut":
      raise DefinitionError(
        f'Class {name} is missing the required method(s): '
        f'{", ".join(missing_methods)}')

    # Otherwise, saving the class
    if name != 'InOut':
      cls.classes[name] = cls


class InOut(metaclass=MetaIO):
  """Base class for all InOut objects. Implements methods shared by all the
  these objects, and ensures their dataclass is MetaIO."""

  def get_data(self) -> Optional[Union[list, Dict[str, Any]]]:
    """"""

    print(f"WARNING ! The InOut {type(self).__name__} has downstream links but"
          f" its get_data method is not defined !\nNo data sent to downstream "
          f"links.")
    sleep(1)
    return

  def set_cmd(self, *_, **__) -> None:
    """"""

    print(f"WARNING ! The InOut {type(self).__name__} has incoming links but"
          f" its set_cmd method is not defined !\nThe data received from the "
          f"incoming links is discarded.")
    sleep(1)
    return

  def start_stream(self) -> None:
    """"""

    print(f"WARNING ! The InOut {type(self).__name__} does not define the "
          f"start_stream method !")

  def get_stream(self) -> Optional[Union[list, Dict[str, Any]]]:
    """"""

    print(f"WARNING ! The InOut {type(self).__name__} has downstream links but"
          f" its get_stream method is not defined !\nNo data sent to "
          f"downstream links.")
    sleep(1)
    return

  def stop_stream(self) -> None:
    """"""

    print(f"WARNING ! The InOut {type(self).__name__} does not define the "
          f"stop_stream method !")

  def eval_offset(self, delay: float = 2) -> list:
    """Method formerly used for offsetting the output of an InOut object.
    Kept for backwards-compatibility reason only.

    Acquires data for a given delay and returns for each label the opposite of
    the average of the acquired values. It is then up to the user to use this
    output to offset the data.
    """

    if not hasattr(self, 'get_data'):
      raise IOError("The eval_offset method cannot be called by an InOut that "
                    "doesn't implement the get_data method.")

    t0 = time()
    buf = []

    # Acquiring data for a given delay
    while time() < t0 + delay:
      buf.append(self.get_data()[1:])

    # Averaging the acquired values
    ret = []
    for label_values in zip(*buf):
      ret.append(-sum(label_values) / len(label_values))

    return ret
