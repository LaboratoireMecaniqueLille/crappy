# coding:utf-8

from typing import NoReturn, Any, TYPE_CHECKING as _TYPE_CHECKING
from types import ModuleType
from importlib import import_module
import webbrowser

from . import _exceptions

if _TYPE_CHECKING:
  from ._resources import resources

__all__ = ('OptionalModule', 'docs', 'resources')


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


def __getattr__(name: str) -> Any:
  """Loads the resource namespace on first access."""

  if name == 'resources':
    from ._resources import resources
    globals()[name] = resources
    return resources

  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  """Includes lazy exports without loading them."""

  return sorted(set(globals()) | set(__all__))
