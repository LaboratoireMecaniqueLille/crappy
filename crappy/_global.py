# coding:utf-8

from typing import Optional, NoReturn


class OptionalModule:
  """Placeholder for optional dependencies when not installed

  Will display a message and raise an error when trying to use them
  """

  def __init__(self, module_name: str, message: Optional[str] = None):
    """"""

    self.mname = module_name
    if message is not None:
      self.message = message
    else:
      self.message = f"The module {self.mname} is necessary to use this " \
                     f"functionality. Please install it and try again"

  def __getattr__(self, _) -> NoReturn:
    """"""

    raise RuntimeError(f"Missing module: {self.mname}\n{self.message}")

  def __call__(self, *_, **__) -> NoReturn:
    """"""

    raise RuntimeError(f"Missing module: {self.mname}\n{self.message}")


class LinkDataError(ValueError):
  """Error to raise when trying to send a wrong data type through a Link."""


class StartTimeout(TimeoutError):
  """Exception raised when the start event takes too long to be set."""


class PrepareError(IOError):
  """Error raised in a Block when waiting for all Blocks to be ready but
  another Block fails to prepare."""


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
