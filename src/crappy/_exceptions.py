# coding: utf-8

"""Exceptions used for expected stops, remote failures and Process failures."""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
  from ._failure import FailureReport


class StopRequested(Exception):
  """Signals that a framework operation was stopped on purpose.

  This is a normal stop, not a failure in the final exception group.

  .. versionadded:: 2.2.0
  """


class OptionalDependencyError(ImportError):
  """An optional dependency could not be loaded.

  The original import error is available as the cause. The dependency may be
  installed but unusable, for example because a required library is missing.

  .. versionadded:: 2.2.0
  """

  def __init__(self,
               import_name: str,
               install_name: str | None = None,
               message: str | None = None) -> None:
    """Describes a dependency that could not be loaded.

    Args:
      import_name: Python module or dependency requested by the code.
      install_name: Package name to suggest for installation. :obj:`None` omits
        installation guidance.
      message: Extra guidance shown with the error. :obj:`None` adds no extra
        message.
    """

    self.import_name: str = import_name
    self.install_name: str | None = install_name
    self.message: str | None = message
    diagnostic = f"Optional dependency {import_name!r} could not be loaded."
    if install_name is not None:
      diagnostic += (f" Check the installation of {install_name!r} and its "
                     f"dependencies.")
    if message is not None:
      diagnostic += f"\n{message}"

    super().__init__(diagnostic, name=import_name)

  def __reduce__(self) -> tuple[type, tuple[Any, ...], dict[str, Any]]:
    """Describes how to save and restore this exception.

    Returns:
      Exception class, constructor arguments, and saved attributes.
    """

    return (type(self), (self.import_name, self.install_name, self.message),
            self.__dict__.copy())


class RemoteFailureMetadata:
  """Provides the original details of a failure from another process.

  These details remain available when the original exception type cannot
  be restored.

  Attributes:
    report: :class:`~crappy._failure.FailureReport` describing the failure.
    node_index: Position of this exception in the report's ``nodes``.
    node: :class:`~crappy._failure.ExceptionNode` describing this exception.
    reconstruction_error: Explanation of why the original exception
      could not be restored.

  .. versionadded:: 2.2.0
  """

  def __init__(self,
               report: FailureReport,
               node_index: int,
               reconstruction_error: str) -> None:
    """Stores the report and the selected Exception's details.

    Args:
      report: Report containing the original Exception's details.
      node_index: Index of that Exception in report.nodes.
      reconstruction_error: Explanation of why restoring the original
        Exception failed.
    """

    if (type(node_index) is not int or
        not 0 <= node_index < len(report.nodes)):
      raise ValueError("node_index must reference a node in the report")
    if type(reconstruction_error) is not str:
      raise TypeError("reconstruction_error must be a string")

    self.report = report
    self.node_index = node_index
    self.node = report.nodes[node_index]
    self.reconstruction_error = reconstruction_error

    super().__init__(f"{self.node.module}.{self.node.type_name}: "
                     f"{self.node.message}")

  @staticmethod
  def from_report(report: FailureReport,
                  node_index: int,
                  reconstruction_error: str
                  ) -> (RemoteFailure | RemoteKeyboardInterrupt |
                        RemoteSystemExit | RemoteBaseFailure):
    """Creates a replacement for an exception from another process.

    Args:
      report: :class:`~crappy._failure.FailureReport` describing the failure.
      node_index: Position of the exception to replace in the report's
        ``nodes``. This must describe an individual exception, not a group.
      reconstruction_error: Explanation of why the original exception
        could not be restored.

    Returns:
      :exc:`RemoteKeyboardInterrupt` for an interrupt,
      :exc:`RemoteSystemExit` for an exit request, :exc:`RemoteFailure` for an
      ordinary exception, or :exc:`RemoteBaseFailure` for another
      :exc:`BaseException`.

    Raises:
      ValueError: The position is invalid or describes an exception group.
    """

    if (type(node_index) is not int or
        not 0 <= node_index < len(report.nodes)):
      raise ValueError("node_index must reference a node in the report")

    # Cannot build a single Exception from an ExceptionGroup
    node = report.nodes[node_index]
    if (node.group_message is not None or 'builtins.BaseExceptionGroup'
        in node.mro):
      raise ValueError("Exception groups must be rebuilt from their children")

    # Select the type of Exception to create based on the MRO of the node
    if 'builtins.KeyboardInterrupt' in node.mro:
      error_type = RemoteKeyboardInterrupt
    elif 'builtins.SystemExit' in node.mro:
      error_type = RemoteSystemExit
    elif 'builtins.Exception' in node.mro:
      error_type = RemoteFailure
    else:
      error_type = RemoteBaseFailure

    # Build and return the Exception
    return error_type(report, node_index, reconstruction_error)

  def __reduce__(self) -> tuple[type, tuple[Any, ...], dict[str, Any]]:
    """Describes how to save and restore the original failure details.

    Returns:
      Exception class, constructor arguments, and saved attributes.
    """

    return (type(self),
            (self.report, self.node_index, self.reconstruction_error),
            self.__dict__.copy())


class RemoteFailure(RemoteFailureMetadata, Exception):
  """Replaces an ordinary exception whose original type cannot be restored.

  Original failure details are available through
  :class:`RemoteFailureMetadata`.

  .. versionadded:: 2.2.0
  """


class RemoteKeyboardInterrupt(RemoteFailureMetadata, KeyboardInterrupt):
  """Replaces a :exc:`KeyboardInterrupt` from another process.

  Still matches ``except* KeyboardInterrupt`` when its original type cannot
  be restored.

  .. versionadded:: 2.2.0
  """


class RemoteSystemExit(RemoteFailureMetadata, SystemExit):
  """Replaces a :exc:`SystemExit` from another process.

  Attributes:
    code: Original exit code, or :obj:`None` if it could not be retained.

  .. versionadded:: 2.2.0
  """

  def __init__(self,
               report: FailureReport,
               node_index: int,
               reconstruction_error: str) -> None:
    """Stores the failure details and the original exit code.

    Args:
      report: Report containing the original SystemExit's details.
      node_index: Index of that Exception in report.nodes.
      reconstruction_error: Explanation of why restoring the original
        Exception failed.
    """

    super().__init__(report, node_index, reconstruction_error)
    self.code = self.node.system_exit_code


class RemoteBaseFailure(RemoteFailureMetadata, BaseException):
  """Replaces another :exc:`BaseException` from another process.

  Use :class:`RemoteFailureMetadata` to read the original failure details.

  .. versionadded:: 2.2.0
  """


class ProcessExitError(RuntimeError):
  """Reports a process that exited unexpectedly.

  Used when the process outcome is not explained by a reported exception.

  .. versionadded:: 2.2.0
  """

  def __init__(self,
               message: str,
               process_name: str,
               pid: int | None,
               phase: str,
               exitcode: int | None) -> None:
    """Describes the process that exited unexpectedly.

    Args:
      message: Description of the unexpected exit.
      process_name: Name of the process that exited.
      pid: Process ID, or :obj:`None` if unknown.
      phase: Last known stage, such as ``prepare`` or ``main``.
      exitcode: Process exit status: zero for a normal exit, a positive value
        for an error, or a negative signal number on POSIX.
        :obj:`None` means the status is not available.
    """

    super().__init__(message)
    self.process_name = process_name
    self.pid = pid
    self.phase = phase
    self.exitcode = exitcode

  def __reduce__(self) -> tuple[type, tuple[Any, ...], dict[str, Any]]:
    """Describes how to save and restore this exception.

    Returns:
      Exception class, constructor arguments, and saved attributes.
    """

    return (type(self), (self.args[0], self.process_name, self.pid, self.phase,
                         self.exitcode), self.__dict__.copy())


class ShutdownTimeoutError(TimeoutError):
  """Reports a process or thread that did not stop within the allowed time.

  .. versionadded:: 2.2.0
  """

  def __init__(self,
               message: str,
               process_name: str = '',
               pid: int | None = None,
               phase: str = '',
               timeout: float | None = None) -> None:
    """Describes a shutdown that took too long.

    Args:
      message: Description of the shutdown failure.
      process_name: Name of the affected process. Empty if unknown.
      pid: Process ID, or :obj:`None` if unknown.
      phase: Stage during shutdown. Empty if unknown.
      timeout: Time allowed for shutdown in seconds.
        :obj:`None` means the allowance is unknown.
    """

    super().__init__(message)
    self.process_name = process_name
    self.pid = pid
    self.phase = phase
    self.timeout = timeout


class FailureTransportError(RuntimeError):
  """Reports a problem sending, receiving or reading a failure report.

  .. versionadded:: 2.2.0
  """

  def __init__(self,
               message: str,
               failure_id: str | None = None,
               process_name: str = '',
               pid: int | None = None,
               phase: str = '') -> None:
    """Describes a problem handling a failure report.

    Args:
      message: Description of the reporting error.
      failure_id: Affected report's identifier, or :obj:`None` if unknown.
      process_name: Name of the affected process. Empty if unknown.
      pid: Process ID, or :obj:`None` if unknown.
      phase: Stage of the affected failure. Empty if unknown.
    """

    super().__init__(message)
    self.failure_id = failure_id
    self.process_name = process_name
    self.pid = pid
    self.phase = phase


class LinkDataError(ValueError):
  """Exception raised when trying to send a wrong data type through a
  :class:`~crappy.links.link.Link`."""


class StartTimeout(TimeoutError):
  """Exception raised when the start event takes too long to be set."""


class T0NotSetError(ValueError):
  """Exception raised when requesting the t0 value when it is not set."""


class DefinitionError(NameError):
  """Raised when defining an object whose name is already registered."""

  def __init__(self, msg: str) -> None:
    """Stores the definition error's message.

    Args:
      msg: Message explaining the conflicting definition.
    """

    super().__init__(msg)
    self._msg = msg

  def __str__(self) -> str:
    """Returns the explanation of the conflicting definition.

    Returns:
      Error message supplied when the exception was created.
    """

    return self._msg


class GeneratorStop(Exception):
  """Exception raised when a :class:`~crappy.blocks.Generator` Block reaches
  the end of its :class:`~crappy.blocks.generator_path.meta_path.path.Path`."""


class SchedulerStop(Exception):
  """Exception raised when a :class:`~crappy.blocks.Scheduler` Block reaches
  its end State."""


class ReaderStop(Exception):
  """Exception raised when a :class:`~crappy.camera.FileReader` Camera has
  exhausted all the images to read."""
