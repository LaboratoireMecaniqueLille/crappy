# coding: utf-8

"""Exceptions used for expected stops, remote failures and Process failures."""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
  from ._failure import FailureReport


class StopRequested(Exception):
  """Signals that a framework operation was stopped on purpose.

  Not included in the final failure report.
  """


class OptionalDependencyError(ImportError):
  """An optional dependency could not be loaded.

  The original import failure is attached as the cause. The dependency may be
  installed but unusable, for example because one of its dependencies is absent
  or a compiled extension cannot be loaded.
  """

  def __init__(self,
               import_name: str,
               install_name: str | None = None,
               message: str | None = None) -> None:
    """Stores information on the import failure.

    Args:
      import_name: Python module or requested dependency name.
      install_name: Package name used for installation, if known.
      message: Additional guidance for the requested functionality.
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

  def __reduce__(self) -> tuple[Any, ...]:
    """Helper for good serialization of this Exception."""

    return (type(self), (self.import_name, self.install_name, self.message),
            self.__dict__.copy())


class RemoteFailureMetadata:
  """Adds the original failure details to a remote fallback Exception.

  Used when the original Exception cannot be restored in this Process.
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
    """Builds a fallback Exception for one node in the report.

    Uses the original class's base classes to choose the fallback type.
    ExceptionGroups must be rebuilt from their children instead.

    Args:
      report: Report containing the original Exception's details.
      node_index: Index of the Exception to replace. Must not refer to a group.
      reconstruction_error: Explanation of why restoring the original
        Exception failed.

    Returns:
      RemoteFailure, RemoteKeyboardInterrupt, RemoteSystemExit or
      RemoteBaseFailure.
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

  def __reduce__(self) -> tuple[Any, ...]:
    """Tells pickle how to restore this Exception's report and attributes."""

    return (type(self),
            (self.report, self.node_index, self.reconstruction_error),
            self.__dict__.copy())


class RemoteFailure(RemoteFailureMetadata, Exception):
  """Fallback for an ordinary remote Exception that cannot be restored."""


class RemoteKeyboardInterrupt(RemoteFailureMetadata, KeyboardInterrupt):
  """Fallback for a remote KeyboardInterrupt that cannot be restored."""


class RemoteSystemExit(RemoteFailureMetadata, SystemExit):
  """Fallback for a remote SystemExit that cannot be restored.

  Attributes:
    code: Original exit code saved in the selected ExceptionNode.
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
  """Fallback for another remote BaseException that cannot be restored."""


class ProcessExitError(RuntimeError):
  """Raised when a Process exits unexpectedly without a reported Exception."""

  def __init__(self,
               message: str,
               process_name: str,
               pid: int | None,
               phase: str,
               exitcode: int | None) -> None:
    """Stores details of the unexpected Process exit.

    Args:
      message: Description of the unexpected exit.
      process_name: Name of the Process that exited.
      pid: ID of that Process. None if unknown.
      phase: Last known lifecycle phase.
      exitcode: Process.exitcode value. None if it is not yet available.
    """

    super().__init__(message)
    self.process_name = process_name
    self.pid = pid
    self.phase = phase
    self.exitcode = exitcode

  def __reduce__(self) -> tuple[Any, ...]:
    """Helper for good serialization of this Exception."""

    return (type(self), (self.args[0], self.process_name, self.pid, self.phase,
                         self.exitcode), self.__dict__.copy())


class ShutdownTimeoutError(TimeoutError):
  """Raised when a Process does not stop within the allowed time."""

  def __init__(self,
               message: str,
               process_name: str = '',
               pid: int | None = None,
               phase: str = '',
               timeout: float | None = None) -> None:
    """Stores details of the Process that did not stop in time.

    Args:
      message: Description of the shutdown failure.
      process_name: Name of the Process that did not stop. Empty if unknown.
      pid: ID of that Process. None if unknown.
      phase: Lifecycle phase during shutdown. Empty if unknown.
      timeout: Time allowed for shutdown, in seconds. None if unknown.
    """

    super().__init__(message)
    self.process_name = process_name
    self.pid = pid
    self.phase = phase
    self.timeout = timeout


class FailureTransportError(RuntimeError):
  """Describes an error in handling or transferring a failure report."""

  def __init__(self,
               message: str,
               failure_id: str | None = None,
               process_name: str = '',
               pid: int | None = None,
               phase: str = '') -> None:
    """Stores details of the failure report that could not be handled.

    Args:
      message: Description of the reporting or transfer error.
      failure_id: ID of the affected report. None if unknown.
      process_name: Name of the affected Process. Empty if unknown.
      pid: ID of that Process. None if unknown.
      phase: Lifecycle phase of the affected failure. Empty if unknown.
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
