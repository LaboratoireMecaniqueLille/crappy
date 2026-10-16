# coding:utf-8

from typing import NoReturn, Any
from importlib import import_module
import webbrowser
import importlib.resources
from numpy import frombuffer, uint8

from . import _exceptions


# Quick access to documentation
def docs():
  """Opens the online documentation of Crappy.

  It opens the latest version, and of course requires an internet access.

  .. versionadded:: 1.5.5
  .. versionchanged:: 2.0.0 renamed from *doc* to *docs*
  """

  webbrowser.open('https://crappy.readthedocs.io/en/latest/')


class OptionalModule:
  """Placeholder for an unavailable or lazily imported optional dependency.

  Reports an import failure only when the dependency is used. It retains the
  original exception as its cause when available.

  .. versionadded:: 1.4.0
  """

  def __init__(self,
               module_name: str,
               message: str | None = None,
               lazy_import: bool = False,
               cause: ImportError | None = None,
               install_name: str | None = None) -> None:
    """Sets the arguments.

    Args:
      module_name: Python import name, such as ``cv2`` or ``PIL``.
      message: Additional guidance displayed if the dependency cannot load.
      lazy_import: If :obj:`True`, the module won't be imported directly even
        if it is installed. Instead, it will be imported only when necessary.
        Allows reducing the import time, especially on Windows.

        .. versionadded:: 2.0.0
      cause: Original failure from an import, if available.
      install_name: Installation package name when different from the import
        name, such as ``opencv-python`` or ``Pillow``.
    """

    self._name: str = module_name
    self._msg: str | None = message
    self._cause: ImportError | None = cause
    self._install_name: str = install_name or module_name.split('.')[0]
    self._lazy: bool = lazy_import
    self._module = None

  def _resolve(self) -> Any:
    """Imports a lazy dependency or reports why its import failed."""

    # A non-lazy OptionalModule must fail when resolved
    if not self._lazy:
      self._raise_unavailable()

    # Trying to import the module
    if self._module is None:
      try:
        self._module = import_module(self._name)
      except ImportError as error:
        self._cause = error
        self._raise_unavailable()
      self._cause = None

    # If import succeeded, just return the module
    return self._module

  def _raise_unavailable(self) -> NoReturn:
    """Raises an OptionalDependencyError while preserving the context."""

    raise _exceptions.OptionalDependencyError(self._name,
                                              self._install_name,
                                              self._msg) from self._cause

  def __getattr__(self, attr: str) -> Any:
    """Method normally raising an exception indicating that the module is
    missing.

    In case the ``lazy_import`` argument was set to :obj:`True`, still tries to
    get the desired attribute and raises the exception only if the module is
    missing.
    """

    return getattr(self._resolve(), attr)

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Method raising an exception indicating that the module is missing.

    In case the ``lazy_import`` argument was set to :obj:`True`, still tries to
    call the desired method and raises the exception only if the module is
    missing.
    """

    return self._resolve()(*args, **kwargs)


# Data aliases
class resources:
  """This class defines aliases for quick access to the resources in the
  `tool/data/` folder.

  These aliases are then used in the examples provided on the GitHub
  repository, but could also be used in custom user scripts.

  .. versionadded:: 1.5.3
  """

  try:
    # Defining aliases to the images themselves
    from cv2 import imdecode
    speckle_path = importlib.resources.files('crappy').joinpath(
        'tool/data/speckle.png')
    speckle = imdecode(frombuffer(speckle_path.read_bytes(), uint8),
                       flags=0)

    ve_markers_path = importlib.resources.files('crappy').joinpath(
        'tool/data/ve_markers.tif')
    ve_markers = imdecode(frombuffer(ve_markers_path.read_bytes(), uint8),
                          flags=0)

    pad_path = importlib.resources.files('crappy').joinpath(
        'tool/data/pad.png')
    pad = imdecode(frombuffer(pad_path.read_bytes(), uint8),
                   flags=0)

  # In case the module opencv-python is missing
  except (ModuleNotFoundError, ImportError):
    speckle = OptionalModule('opencv-python')
    ve_markers = OptionalModule('opencv-python')
    pad = OptionalModule('opencv-python')

  # Also getting the paths to the images
  ref_pad = (importlib.resources.files('crappy') / 'tool' / 'data' /
             'pad.png')
  ref_speckle = (importlib.resources.files('crappy') / 'tool' / 'data' /
                 'speckle.png')
  ref_ve = (importlib.resources.files('crappy') / 'tool' / 'data' /
            've_markers.tif')
  with (importlib.resources.as_file(ref_pad) as path_pad,
        importlib.resources.as_file(ref_speckle) as path_speckle,
        importlib.resources.as_file(ref_ve) as path_ve):
    paths = {'pad': path_pad, 'speckle': path_speckle, 've_markers': path_ve}
