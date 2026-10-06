# coding: utf-8

from dataclasses import dataclass
from importlib import import_module
from typing import Literal


@dataclass(frozen=True, slots=True)
class CollectionEntry:
  """Minimal information required by Crappy to load a collection class.

  Args:
    name: The name of the class to load, as a :obj:`str` (e.g. 'MCP9600').
    kind: The type of hardware the class is meant to drive, must be one of
      'Actuator', 'Camera', 'InOut'.
    module: The fully resolved module containing the class to load, as a
      :obj:`str` (e.g. 'crappy.collection.inout.mcp9600').
  """

  name: str
  kind: Literal['Actuator', 'Camera', 'InOut']
  module: str

  def __post_init__(self) -> None:
    """Checks the validity of the provided arguments."""

    if not isinstance(self.name, str) or not self.name:
      raise ValueError("name must be a non-empty string")
    if self.kind not in ('Actuator', 'Camera', 'InOut'):
      raise ValueError("kind must be one of 'Actuator', 'Camera', InOut'")
    if not isinstance(self.module, str) or not self.module:
      raise ValueError("module must be a non-empty string")


class CollectionUnavailableError(RuntimeError):
  """Raised when a declared collection class cannot be loaded."""

  def __init__(self,
               entry: CollectionEntry,
               cause: Exception) -> None:
    """Raises a :exc:`RuntimeError` with a custom message.

    Args:
      entry: The CollectionEntry that failed to load.
      cause: The Exception that caused the loading failure.
    """

    if not isinstance(entry, CollectionEntry):
      raise TypeError("The provided entry must be a CollectionEntry")
    if not isinstance(cause, Exception):
      raise TypeError("The provided cause must be an Exception")

    self.entry = entry
    self.cause = cause

    super().__init__(f"The {entry.kind} {entry.name!r} is declared in "
                     f"crappy.collection, but could not be loaded.\n"
                     f"Importing {entry.module!r} raised "
                     f"{type(cause).__name__}: {cause}")


class CollectionRegistry:
  """Registry of all the classes declared by `crappy.collection`.

  This registry is empty until `crappy.collection` is imported and populates
  it.
  """

  def __init__(self) -> None:
    """Initializes the internal registry."""

    self._entries: dict[tuple[str, str], CollectionEntry] = dict()

  def register(self, *entries: CollectionEntry) -> None:
    """Registers collection entries.

    Args:
      *entries: All the entries to register, which are simply added to the
        internal registry.
    """

    for entry in entries:
      if not isinstance(entry, CollectionEntry):
        raise TypeError(f"The entry {entry} isn't a CollectionEntry!")

      # Store this pair to allow identical names for objects of different types
      key = (entry.kind, entry.name)

      # Re-registering exactly the same entry is harmless
      if key in self._entries:
        # But overwriting an existing entry is not allowed (detects duplicates)
        if self._entries[key] != entry:
          raise RuntimeError(f"Conflicting collection entries for "
                             f"{entry.kind} {entry.name!r}")
        continue

      self._entries[key] = entry

  def get(self,
          kind: Literal['Actuator', 'Camera', 'InOut'],
          name: str) -> CollectionEntry | None:
    """Returns the declared entry of the requested kind and with the requested
    name, if one is registered.

    Args:
      kind: The type of hardware the requested class drives, must be one of
       'Actuator', 'Camera', 'InOut'.
      name: The name of the requested class, as a :obj:`str`.
    """

    if kind not in ('Actuator', 'Camera', 'InOut'):
      raise ValueError("kind must be one of 'Actuator', 'Camera', InOut'")
    if not isinstance(name, str) or not name:
      raise ValueError("name must be a non-empty string")

    return self._entries.get((kind, name))

  def find(self, name: str) -> tuple[CollectionEntry, ...]:
    """Returns all the entries matching a name.

    Args:
      name: The name of the requested class, as a :obj:`str`.

    Returns:
      All the CollectionEntry with the requested name. Since names must be
      unique for a given hardware kind, and there are three kinds of hardware,
      the returned :obj:`tuple` contains 0 to 3 values.
    """

    if not isinstance(name, str) or not name:
      raise ValueError("name must be a non-empty string")

    return tuple(entry for entry in self._entries.values()
                 if entry.name == name)

  def entries(self,
              kind: str | None = None) -> tuple[CollectionEntry, ...]:
    """Returns all the registered collection entries of the requested kind.

    Args:
      kind: If not :obj:`None`, must be one of 'Actuator', 'Camera', 'InOut'.
        Only the entries declared with this exact kind will be returned.

    Returns:
      A :obj:`tuple` containing all the registered CollectionEntry matching the
      provided kind, or just all the CollectionEntry if kind is :obj:`None`.
    """

    if kind is not None and kind not in ('Actuator', 'Camera', 'InOut'):
      raise ValueError("kind must be one of 'Actuator', 'Camera', InOut'")

    entries = (entry for entry in self._entries.values()
               if kind is None or entry.kind == kind)

    return tuple(sorted(entries,
                        key=lambda entry: (entry.kind, entry.name)))


def load_collection_class(entry: CollectionEntry,
                          crappy_dict: dict[str, type]) -> type:
  """Imports and validates a class declared in `crappy.collection`.

  Args:
    entry: The CollectionEntry containing the description of the class to load.
    crappy_dict: The dictionary in which the loaded class will be added if it
      is successfully imported by Crappy (one of Actuator.classes,
      Camera.classes, InOut.classes).

  Returns:
    The class that was successfully loaded.

  Raises:
    CollectionUnavailableError: If importing the module fails, if the class
      does not register itself, or if another class with the same name is
      registered instead.
  """

  try:
    import_module(entry.module)

  # Collection code is allowed to fail for any reason, reporting the failure
  except Exception as exc:
    raise CollectionUnavailableError(entry, exc) from exc

  # Check that the class to load was added to Crappy's internal register
  cls = crappy_dict.get(entry.name)
  if cls is None:
    exc = RuntimeError(f"The module imported successfully but did not "
                       f"register {entry.name!r}")
    raise CollectionUnavailableError(entry, exc) from exc

  # Check that the module containing the loaded class is the expected one
  if cls.__module__ != entry.module:
    exc = RuntimeError(f"{entry.name!r} is registered by {cls.__module__!r}, "
                       f"but the collection entry declares {entry.module!r}")
    raise CollectionUnavailableError(entry, exc) from exc

  return cls


# The public object storing the CollectionRegistry, accessible by Crappy Blocks
collection_registry: CollectionRegistry = CollectionRegistry()
