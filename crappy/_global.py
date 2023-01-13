# coding:utf-8

from typing import Optional, NoReturn
from importlib import import_module


class OptionalModule:
  """Placeholder for optional dependencies when not installed

  Will display a message and raise an error when trying to use them
  """

  def __init__(self,
               module_name: str,
               message: Optional[str] = None,
               lazy_import: bool = False):
    """"""

    self._name = module_name

    # Setting a default message if none was provided
    if message is not None:
      self._msg = message
    else:
      self._msg = f"The module {self._name} is necessary to use this " \
                  f"functionality. Please install it and try again"

    self._lazy = lazy_import
    self._module = None

  def __getattr__(self, attr) -> NoReturn:
    """"""

    # The module has to be imported only when called because it's too heavy
    if self._lazy:
      raise_ = False

      # Trying to import the module
      if self._module is None:
        try:
          self._module = import_module(self._name)
        except (ImportError, ModuleNotFoundError):
          raise_ = True

      # The module could be imported, returning the desired attribute
      if not raise_:
        return getattr(self._module, attr)

    # The module could not be imported
    raise RuntimeError(f"Missing module: {self._name}\n{self._msg}")

  def __call__(self, *_, **__) -> NoReturn:
    """"""

    raise RuntimeError(f"Missing module: {self._name}\n{self._msg}")


class LinkDataError(ValueError):
  """Error to raise when trying to send a wrong data type through a Link."""


class StartTimeout(TimeoutError):
  """Exception raised when the start event takes too long to be set."""


class PrepareError(IOError):
  """Error raised in a Block when waiting for all Blocks to be ready but
  another Block fails to prepare."""


class CameraPrepareError(RuntimeError):
  """Error raised by a Camera Block when one of its child processes crashes
  while preparing."""


class CameraRuntimeError(RuntimeError):
  """Error raised by a Camera Block when one of its child processes crashes
  while running."""


class T0NotSetError(ValueError):
  """Exception raised when requesting the t0 value when it is not set."""


class DefinitionError(Exception):
    """Error to raise when classes are not defined correctly"""

    def __init__(self, msg=""):
      super().__init__()
      self.msg = msg

    def __str__(self):
      return self.msg


class GeneratorStop(Exception):
  """Exception raised when a Generator block reaches the end of its path."""


class ReaderStop(Exception):
  """Exception raised when a File_reader camera has exhausted all the images
  to read."""
