# coding:utf-8

from typing import NoReturn, Any
from types import ModuleType
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
    """Describes an optional dependency and how it should be loaded.

    Args:
      module_name: Python import name, such as ``cv2`` or ``PIL``.
      message: Extra guidance shown if the dependency cannot load. :obj:`None`
        adds no extra message.
      lazy_import: If :obj:`True`, try to load the dependency when it is first
        used. If :obj:`False`, using this placeholder reports the dependency
        as unavailable.

        .. versionadded:: 2.0.0
      cause: Original import error, or :obj:`None` if unavailable.
      install_name: Installation package name, such as ``opencv-python`` or
        ``Pillow``. :obj:`None` uses the first part of ``module_name``.

    .. versionchanged:: 2.2.0
    """

    self._name: str = module_name
    self._msg: str | None = message
    self._cause: ImportError | None = cause
    self._install_name: str = install_name or module_name.split('.')[0]
    self._lazy: bool = lazy_import
    self._module: ModuleType | None = None

  def is_available(self) -> bool:
    """Checks whether the dependency can be loaded.

    Lazy dependencies are imported and successful imports are cached. Import
    failures return :obj:`False` and can be retried on a later check or use.
    Other initialization errors and interruptions propagate unchanged.
    Non-lazy placeholders return :obj:`False` without importing anything.

    Returns:
      Whether the dependency could be loaded.

    .. versionadded:: 2.2.0
    """

    try:
      self.resolve()
    except _exceptions.OptionalDependencyError:
      return False
    return True

  def resolve(self) -> ModuleType:
    """Loads the dependency and returns the cached module.

    Failed lazy imports can be retried by a later call. Non-lazy placeholders
    report the dependency as unavailable without trying to import it.

    Returns:
      Imported module.

    Raises:
      ~crappy._exceptions.OptionalDependencyError: The dependency is
        unavailable. Its original import error is retained as the cause.

    .. versionadded:: 2.2.0
    """

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
    """Raises an OptionalDependencyError while preserving the context.

    .. versionadded:: 2.2.0
    """

    raise _exceptions.OptionalDependencyError(self._name,
                                              self._install_name,
                                              self._msg) from self._cause

  def __getattr__(self, attr: str) -> Any:
    """Returns an attribute from a lazily loaded dependency.

    Args:
      attr: Name of the requested attribute.

    Returns:
      Attribute from the imported module.

    Raises:
      ~crappy._exceptions.OptionalDependencyError: The dependency is
        unavailable.
      AttributeError: The loaded module has no such attribute.
    """

    return getattr(self.resolve(), attr)

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Calls the loaded dependency if it is callable.

    Args:
      *args: Positional arguments passed to the callable.
      **kwargs: Keyword arguments passed to the callable.

    Returns:
      Object returned by the callable.

    Raises:
      ~crappy._exceptions.OptionalDependencyError: The dependency is u
        navailable.
      TypeError: The loaded dependency is not callable.
    """

    return self.resolve()(*args, **kwargs)


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
  except (ModuleNotFoundError, ImportError) as error:
    speckle = OptionalModule('cv2', cause=error, install_name='opencv-python')
    ve_markers = OptionalModule('cv2', cause=error,
                                install_name='opencv-python')
    pad = OptionalModule('cv2', cause=error, install_name='opencv-python')

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
