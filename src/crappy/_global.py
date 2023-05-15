# coding:utf-8

from typing import Optional, NoReturn, Any
from importlib import import_module


class OptionalModule:
  """Placeholder for optional dependencies that are not installed.

  Will display a message and raise an error when trying to use them.
  """

  def __init__(self,
               module_name: str,
               message: Optional[str] = None,
               lazy_import: bool = False) -> None:
    """Sets the arguments.

    Args:
      module_name: The name of the module s a :obj:`str`, preferably the one
        invoked with `pip install`.
      message: Optionally, the message to display in case the module is
        missing (as a :obj:`str`). If not provided, a generic message will be
        displayed.
      lazy_import: If :obj:`True`, the module won't be imported directly even
        if it is installed. In stead, it will be imported only when necessary.
        Allows reducing the import time, especially on Window.
    """

    self._name = module_name

    # Setting a default message if none was provided
    if message is not None:
      self._msg = message
    else:
      self._msg = f"The module {self._name} is necessary to use this " \
                  f"functionality. Please install it and try again !"

    self._lazy = lazy_import
    self._module = None

  def __getattr__(self, attr: str) -> Any:
    """Method normally raising an exception indicating that the module is
    missing.

    In case the ``lazy_import`` argument was set to :obj:`True`, still tries to
    get the desired attribute and raises the exception only if the module is
    missing.
    """

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
    """Method raising an exception indicating that the module is missing."""

    raise RuntimeError(f"Missing module: {self._name}\n{self._msg}")


class LinkDataError(ValueError):
  """Exception raised when trying to send a wrong data type through a
  :class:`~crappy.links.Link`."""


class StartTimeout(TimeoutError):
  """Exception raised when the start event takes too long to be set."""


class PrepareError(IOError):
  """Error raised in a :class:`~crappy.blocks.Block` when waiting for all
  Blocks to be ready but another Block fails to prepare."""


class CameraConfigError(RuntimeError):
  """Error raised by a :class:`~crappy.tool.camera_config.CameraConfig` window
  when encountering an exception."""


class CameraPrepareError(RuntimeError):
  """Error raised by a :class:`~crappy.blocks.Camera` when one of its children
  processes crashes while preparing."""


class CameraRuntimeError(RuntimeError):
  """Error raised by a :class:`~crappy.blocks.Camera` when one of its children
  processes crashes while running."""


class T0NotSetError(ValueError):
  """Exception raised when requesting the t0 value when it is not set."""


class DefinitionError(NameError):
    """Exception raised when trying to define an object with the same name as
    an already-defined one."""

    def __init__(self, msg: str) -> None:
      """Sets the msg attribute.

      Args:
        msg: The message to display along with the exception.
      """

      super().__init__()
      self._msg = msg

    def __str__(self) -> str:
      return self._msg


class GeneratorStop(Exception):
  """Exception raised when a :class:`~crappy.blocks.Generator` Block reaches
  the end of its :class:`~crappy.blocks.generator_path.Path`."""


class ReaderStop(Exception):
  """Exception raised when a :class:`~crappy.camera.FileReader` Camera has
  exhausted all the images to read."""
