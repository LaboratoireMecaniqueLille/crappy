# coding: utf-8

from typing import Literal
from dataclasses import dataclass

from .manifest import DRIVERS
from .._collection import (CollectionEntry, CollectionUnavailableError,
                           load_collection_class)


@dataclass(frozen=True, slots=True)
class CheckResult:
  """Result of checking if a collection driver can be loaded.

  Args:
    name: The name of the loaded class, as a :obj:`str` (e.g. 'MCP9600').
    kind: The type of hardware the class is meant to drive, must be one of
      'Actuator', 'Camera', 'InOut'.
    module: The fully resolved module containing the loaded class, as a
      :obj:`str` (e.g. 'crappy.collection.inout.mcp9600').
    available: :obj:`True` if the driver module and class could be loaded.
    error_type: The name of the loading exception type, or :obj:`None` when
      the driver is available.
    error_message: The loading exception message, or :obj:`None` when the
      driver is available.
  """

  name: str
  kind: Literal['Actuator', 'Camera', 'InOut']
  module: str
  available: bool
  error_type: str | None = None
  error_message: str | None = None


def drivers(kind: str | None = None) -> tuple[CollectionEntry, ...]:
  """Lists the drivers declared in `crappy.collection` without importing them.

  Args:
    kind: A specific type of hardware classes to list, must be one of
      'Actuator', 'Camera', 'InOut'. Or :obj:`None` if all the possible types
      of hardware should be listed.

  Returns:
    A :obj:`tuple` containing for each declared driver its CollectionEntry.
  """

  if kind is not None and kind not in ('Actuator', 'Camera', 'InOut'):
    raise ValueError("kind must be one of 'Actuator', 'Camera', InOut'")

  return tuple(driver for driver in DRIVERS if
               (kind is None or driver.kind == kind))


def check(name: str,
          kind: Literal['Actuator', 'Camera', 'InOut'] | None = None
          ) -> CheckResult:
  """Checks whether a given collection driver can be imported, by actually
  trying to import it.

  If the import fails, no error is raised; the exception type and message are
  stored in the returned result.

  Args:
    name: The name of the class to import as a :obj:`str`.
    kind: The type of hardware the class is meant to drive, or :obj:`None` to
      resolve an unambiguous name across 'Actuator', 'Camera', and 'InOut'.

  Returns:
    A CheckResult indicating whether the requested driver is available.
  """

  if not isinstance(name, str) or not name:
    raise ValueError("name must be a non-empty string")
  if kind is not None and kind not in ('Actuator', 'Camera', 'InOut'):
    raise ValueError("kind must be one of 'Actuator', 'Camera', InOut'")

  # Gather the classes matching the request
  matches = tuple(driver for driver in DRIVERS if driver.name == name and
                  (kind is None or driver.kind == kind))

  if not matches:
    raise ValueError(f"No collection driver named {name!r}"
                     f"{'' if kind is None else f' of kind {kind!r}'}")

  # Can happen in case kind is None
  if len(matches) > 1:
    kinds = ", ".join(driver.kind for driver in matches)
    raise ValueError(f"Several collection drivers are named {name!r}, of "
                     f"kinds {kinds}. Specify the kind explicitly.")

  driver, = matches

  # Get the dict in which the driver should be present after loading
  if driver.kind == "Actuator":
    from ..actuator.meta_actuator import Actuator
    crappy_dict = Actuator.classes
  elif driver.kind == "Camera":
    from ..camera.meta_camera import Camera
    crappy_dict = Camera.classes
  elif driver.kind == "InOut":
    from ..inout.meta_inout import InOut
    crappy_dict = InOut.classes
  else:
    raise ValueError("kind must be one of 'Actuator', 'Camera', InOut'")

  # Try to load the module
  try:
    load_collection_class(driver, crappy_dict)
  # If loading fails, indicate it and return the causes
  except CollectionUnavailableError as exc:
    cause = exc.cause
    return CheckResult(name=driver.name,
                       kind=driver.kind,
                       module=driver.module,
                       available=False,
                       error_type=type(cause).__name__,
                       error_message=str(cause))
  # Otherwise indicate that the loading succeeded
  return CheckResult(name=driver.name,
                     kind=driver.kind,
                     module=driver.module,
                     available=True)


def check_all(kind: str | None = None) -> tuple[CheckResult, ...]:
  """Checks the loading of all the `crappy.collection` drivers, by actually
  trying to import them.

  If an import fails, no error is raised; the exception type and message are
  stored in the returned result.

  Args:
    kind: A specific type of hardware classes to check, must be one of
      'Actuator', 'Camera', 'InOut'. Or :obj:`None` if all the possible types
      of hardware should be checked.

  Returns:
    A :obj:`tuple` of CheckResult indicating for each driver whether it is
    available for loading.
  """

  return tuple(check(driver.name, driver.kind) for driver in drivers(kind))
