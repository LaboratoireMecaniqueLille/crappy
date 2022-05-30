# coding: utf-8

from time import time

from .._global import DefinitionError

# Todo: Implement get_data, get_stream and set_cmd in all InOut but returning
#  nothing


class MetaIO(type):
  """Metaclass ensuring that two InOuts don't have the same name, and that all
  InOuts define the required methods. Also keeps track of all the InOut
  classes, including the custom user-defined ones, and sorts them as read_only,
  write-only, or read-write."""

  classes = {}  # All the InOut classes
  IO_classes = {}  # Read-write InOut classes
  O_classes = {}  # Write-only InOut classes
  I_classes = {}  # Read-only InOut classes
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

    # Checking that the class has at least one of the necessary methods
    if name != 'InOut':
      in_flag = ("get_data" in dct or "get_stream" in dct)
      out_flag = ("set_cmd" in dct)
      if in_flag and out_flag:
        cls.IO_classes[name] = cls
      elif in_flag:
        cls.I_classes[name] = cls
      elif out_flag:
        cls.O_classes[name] = cls
      else:
        raise DefinitionError(f'{name} must define at least set_cmd, '
                              f'get_data or get_stream !')
      cls.classes[name] = cls


class InOut(metaclass=MetaIO):
  """Base class for all InOut objects. Implements methods shared by all the
  these objects, and ensures their dataclass is MetaIO."""

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
