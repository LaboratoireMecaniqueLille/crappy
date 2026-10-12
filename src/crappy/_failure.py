# coding: utf-8

"""Classes used for managing failure handling, transfer and reporting."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from multiprocessing import current_process, queues
from os import getpid
from queue import Empty
from threading import RLock, local
from traceback import TracebackException, format_tb
from typing import Any
from uuid import UUID, uuid4
import logging
import pickle
import sys

from ._exceptions import (FailureTransportError, RemoteFailureMetadata,
                          StopRequested)


@dataclass(frozen=True, slots=True)
class ExceptionNode:
  """Stores one Exception's details so they can be sent to another Process.

  Other Exceptions are referenced by their index in FailureReport.nodes.

  Attributes:
    module: Module where the original Exception class is defined.
    type_name: Class name, including any enclosing function or class names.
    message: Text returned by str() on the original Exception.
    mro: Full names of the original class and its base classes, in
      Python's lookup order.
    notes: Messages added with Exception.add_note().
    formatted_stack: Stack frames for this Exception, as formatted text.
    children: Indexes of a group's direct members, in their original order.
      Empty for an Exception that is not a group.
    group_message: Message used to create the group. None for a non-group
      Exception.
    cause: Index of the Exception set with raise ... from ..., if any.
    context: Index of the Exception being handled when this one was raised,
      if any.
    suppress_context: Whether to hide context when displaying the traceback.
      An explicit cause is still displayed.
    original_bytes: Original leaf or custom group prototype saved with pickle,
      if available.
    serialization_error: Explanation of why pickling failed, if it failed.
    system_exit_code: Original SystemExit.code when it is an integer, string,
      boolean or None. Other values are not stored here.
  """

  module: str
  type_name: str
  message: str
  mro: tuple[str, ...] = field()
  notes: tuple[str, ...] = tuple()
  formatted_stack: str = ''
  children: tuple[int, ...] = tuple()
  group_message: str | None = None
  cause: int | None = None
  context: int | None = None
  suppress_context: bool = False
  original_bytes: bytes | None = None
  serialization_error: str | None = None
  system_exit_code: int | str | None = None

  def __post_init__(self) -> None:
    """Checks that the fields contain valid values."""

    for name in ('module', 'type_name', 'message', 'formatted_stack'):
      if type(getattr(self, name)) is not str:
        raise TypeError(f"{name} must be a string")
    if (type(self.mro) is not tuple or
        any(type(name) is not str or not name for name in self.mro)):
      raise TypeError("mro must be an immutable tuple of non-empty type names")
    if (not self.mro or
        self.mro[0] != f"{self.module}.{self.type_name}" or
        'builtins.BaseException' not in self.mro or
        self.mro[-1] != 'builtins.object'):
      raise ValueError("mro must describe the original exception's ancestry")
    if (type(self.notes) is not tuple or
        any(type(note) is not str for note in self.notes)):
      raise TypeError("notes must be an immutable tuple of strings")
    if type(self.children) is not tuple:
      raise TypeError("children must be an immutable tuple of node indexes")
    if any(type(child) is not int or child < 0 for child in self.children):
      raise ValueError("children must contain non-negative integer indexes")
    for reference in (*self.children, self.cause, self.context):
      if (reference is not None and
          (type(reference) is not int or reference < 0)):
        raise ValueError("node references must be non-negative integers")
    if self.group_message is not None and type(self.group_message) is not str:
      raise TypeError("group_message must be a string or None")
    if self.children and self.group_message is None:
      raise ValueError("group children require a group_message")
    if type(self.suppress_context) is not bool:
      raise TypeError("suppress_context must be a boolean")
    if (self.original_bytes is not None and
        type(self.original_bytes) is not bytes):
      raise TypeError("original_bytes must be bytes or None")
    if (self.serialization_error is not None and
        type(self.serialization_error) is not str):
      raise TypeError("serialization_error must be a string or None")
    if type(self.system_exit_code) not in (int, str, bool, type(None)):
      raise TypeError("system_exit_code must be a simple exit code")


@dataclass(frozen=True, slots=True)
class FailureReport:
  """Stores details of an Exception or ExceptionGroup and where it happened.

  Attributes:
    failure_id: UUID written as 32 lowercase hexadecimal characters.
      Kept unchanged when the report is forwarded.
    block_name: Name of the Block or component that owns the failure.
    owner_path: Name showing which Block or worker failed.
    process_name: Name of the Process where the failure happened.
    pid: ID of the Process where the failure happened.
    phase: Lifecycle phase where the failure happened, such as prepare,
      main or finish.
    exception_module: Module of the top-level Exception's class.
    exception_type: Class name of the top-level Exception.
    exception_message: Message of the top-level Exception.
    formatted_traceback: Full traceback text, including group members,
      exception chains and notes.
    nodes: Exception descriptions used for transfer between Processes.
      Empty when the report only stores local diagnostics.
    root: Index of the top-level Exception in nodes. None when nodes is empty.
  """

  failure_id: str
  block_name: str
  owner_path: str
  process_name: str
  pid: int
  phase: str
  exception_module: str
  exception_type: str
  exception_message: str
  formatted_traceback: str
  nodes: tuple[ExceptionNode, ...] = tuple()
  root: int | None = None

  def __post_init__(self) -> None:
    """Checks the field values and references to Exception nodes."""

    if type(self.failure_id) is not str:
      raise TypeError("failure_id must be a UUID hex string")
    if UUID(hex=self.failure_id).hex != self.failure_id:
      raise ValueError("failure_id must be a lowercase UUID hex string")
    for name in ('block_name', 'owner_path', 'process_name', 'phase'):
      if type(getattr(self, name)) is not str or not getattr(self, name):
        raise ValueError(f"{name} must be a non-empty string")
    if type(self.pid) is not int or self.pid <= 0:
      raise ValueError("pid must be a positive integer")
    for name in ('exception_module', 'exception_type', 'exception_message',
                 'formatted_traceback'):
      if type(getattr(self, name)) is not str:
        raise TypeError(f"{name} must be a string")
    if (type(self.nodes) is not tuple or
        any(not isinstance(node, ExceptionNode) for node in self.nodes)):
      raise TypeError("nodes must be an immutable tuple of ExceptionNodes")
    if self.nodes:
      if type(self.root) is not int or not 0 <= self.root < len(self.nodes):
        raise ValueError("root must reference a node in the report")
    elif self.root is not None:
      raise ValueError("a root index requires exception nodes")
    for node in self.nodes:
      for reference in (*node.children, node.cause, node.context):
        if reference is not None and reference >= len(self.nodes):
          raise ValueError("node reference is outside the report")

  def __str__(self) -> str:
    """Returns the source details followed by the full traceback."""

    return (f"Failure {self.failure_id}: Block {self.block_name}, "
            f"owner {self.owner_path}, process {self.process_name} "
            f"(PID {self.pid}), phase {self.phase}\n"
            f"{self.formatted_traceback}")

  def to_wire(self) -> dict[str, Any]:
    """Returns a report containing only built-in values.

    Original Exceptions are already serialized in the nodes. This conversion
    does not run their pickle hooks again.

    Returns:
      Dictionary suitable for transfer between Processes.
    """

    wire = {item.name: getattr(self, item.name) for item in fields(self)}
    wire['nodes'] = list()
    # Add the information of all the nodes in the report
    for node in self.nodes:
      data = {item.name: getattr(node, item.name) for item in fields(node)}
      for name in ('mro', 'notes', 'children'):
        data[name] = list(data[name])
      wire['nodes'].append(data)
    return wire

  @classmethod
  def from_wire(cls, wire: dict[str, Any]) -> FailureReport:
    """Validates and restores a report without reconstructing Exceptions.

    Args:
      wire: Built-in dictionary returned by to_wire().

    Returns:
      Report retaining the original failure ID, source and diagnostic graph.

    Raises:
      FailureTransportError: The fields or wire format are invalid.
    """

    try:
      if type(wire) is not dict:
        raise TypeError("wire must be a dictionary")
      if set(wire) != {item.name for item in fields(cls)}:
        raise ValueError("wire must contain exactly the report fields")
      if type(wire['nodes']) is not list:
        raise TypeError("wire nodes must be a list")

      # Start with reconstruction of all the nodes
      nodes = list()
      for data in wire['nodes']:
        if (type(data) is not dict or
            set(data) != {item.name for item in fields(ExceptionNode)}):
          raise ValueError("wire node must contain exactly the node fields")
        values = data.copy()

        for name in ('mro', 'notes', 'children'):
          if type(values[name]) is not list:
            raise TypeError(f"wire {name} must be a list")
          values[name] = tuple(values[name])

        nodes.append(ExceptionNode(**values))

      # Then reconstruct the report
      values = wire.copy()
      values['nodes'] = tuple(nodes)
      return cls(**values)

    except (TypeError, ValueError) as error:
      raise FailureTransportError(f"Invalid failure report: "
                                  f"{error}") from error


class _CollectionState(local):
  """Stores a separate collection stack for each thread.

  Attributes:
    stack: Active collect() contexts, from outermost to innermost.
      Each entry holds its label, its failures and its already-seen Exceptions.
  """

  stack: list[tuple[str, list[BaseException], dict[int, BaseException]]]

  def __init__(self) -> None:
    """Creates an empty stack for the current thread."""

    self.stack = list()


class FailureManager:
  """Catches, stores, logs and groups failures for one Process.

  capture_exception() records and logs a new failure right away. collect()
  raises the failures from its own context when it exits. Use step() for each
  operation that must run even if an earlier one fails. raise_if_failed()
  raises all the failures kept by the manager.
  """

  def __init__(self,
               block_name: str,
               phase: str,
               owner_path: str | None = None) -> None:
    """Creates an empty manager for the current Process.

    The Process name and PID are read from the current Process.

    Args:
      block_name: Name of the Block or component that owns the failures.
      phase: Default phase for new reports, such as prepare, main or finish.
        Can be updated as execution moves to another phase.
      owner_path: Name showing which Block or worker is being managed. Uses
        block_name when None.
    """

    if owner_path is None:
      owner_path = block_name
    for name, value in (('block_name', block_name), ('phase', phase),
                        ('owner_path', owner_path)):
      if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")

    self.block_name = block_name
    self.owner_path = owner_path
    self.process_name = current_process().name
    self.pid = getpid()
    self.phase = phase

    self._reports: dict[str, tuple[FailureReport, BaseException | None]] = {}
    self._members: dict[int, tuple[BaseException, str]] = dict()
    self._lock = RLock()
    self._collections = _CollectionState()
    self._logger = logging.getLogger('crappy')
    self._logger_failed = False
    self._queue: queues.Queue | None = None
    self._broken_queues: set[queues.Queue] = set()

  def set_queue(self, queue: queues.Queue | None) -> None:
    """Registers a Queue in which to send reports when an Exception is
    registered.

    Args:
      queue: Dedicated Queue received from the immediate owning process.
        Set to None when reporting has finished, before closing the Queue.
    """

    with self._lock:
      self._queue = queue

  def receive(self, incoming: Iterable[queues.Queue]) -> bool:
    """Receives and forwards one available report from each child Queue.

    Args:
      incoming: Queues owned by this process, one per child.

    Returns:
      True if any record was received, including a malformed record.
    """

    # Receive the pending reports from children's queues
    received = False
    for queue in incoming:
      # Ignore queues know to be broken
      if queue in self._broken_queues:
        continue
      # Try to receive a report from the downstream queue
      try:
        payload = queue.get_nowait()
      except Empty:
        continue
      # Remember the broken queues and report the send error
      except BaseException as error:
        self._broken_queues.add(queue)
        self._record_transport_error(error)
        continue

      # Try to interpret the received information as a FailureReport
      received = True
      try:
        if type(payload) is not bytes:
          raise TypeError("failure Queue payload must be bytes")
        report = FailureReport.from_wire(pickle.loads(payload))
      # If this fails, record the associated Exception
      except BaseException as error:
        if (isinstance(error, Exception) and
            not isinstance(error, FailureTransportError)):
          failure = FailureTransportError("Invalid failure Queue payload")
          failure.__cause__ = error
          error = failure
        self._record_transport_error(error)
        continue

      # Pass the received report to the parent Manager, or report any Exception
      try:
        self.forward(report)
      except BaseException as error:
        self._record_transport_error(error)
    return received

  @contextmanager
  def collect(self, label: str) -> Iterator[None]:
    """Collects Exceptions and raises them together when the context exits.

    Use one step() per operation that must be tried independently. An unhandled
    Exception outside step() skips the rest of the context's body. Each
    collection has its own list. All failures also stay in the manager.

    Args:
      label: Collection name added to Exception notes and the group's message.
    """

    if not isinstance(label, str) or not label:
      raise ValueError("label must be a non-empty string")

    # Create the containers for Exceptions
    stack = self._collections.stack
    errors: list[BaseException] = list()
    seen: dict[int, BaseException] = dict()
    stack.append((label, errors, seen))

    try:
      try:
        yield
      # Store any encountered Exception in the stack, generate a report and
      # report the Exception
      except BaseException as error:
        self._record_context_exception(error, 'body')
    # Reset the stack to its original state
    finally:
      stack.pop()

    # Raise all the caught Exceptions on exit
    group = self._make_exception_group(f"{label} failures", errors)
    if group is not None:
      raise group from None

  @contextmanager
  def step(self, label: str) -> Iterator[None]:
    """Catches failures from one operation and allows the next step to run.

    New failures are stored and logged immediately. Must be used inside
    collect().

    Args:
      label: Operation name added to Exception notes.
    """

    if not self._collections.stack:
      raise RuntimeError("step() must be used inside collect()")

    # Store any encountered Exception in the stack, generate a report and
    # report the Exception
    try:
      yield
    except BaseException as error:
      self._record_context_exception(error, label)

  def capture_exception(self,
                        error: BaseException,
                        phase: str | None = None) -> FailureReport | None:
    """Stores, logs and sends a new failure, then returns its report.

    Already reported Exceptions and StopRequested Exceptions are ignored.
    If serialization, logging or sending fails, the original failure
    stays stored alongside that independent error.

    Args:
      error: Exception or ExceptionGroup to record.
      phase: Phase for this report. Uses the manager's current phase when None.

    Returns:
      New FailureReport, or None if no new failure remains.
    """

    # Generate a failure report for the captured Exception
    report, encoding_errors = self._generate_failure_report(error, phase=phase)

    # Immediately report the failure
    reporting_errors = list(encoding_errors)
    if report is not None:
      try:
        self._report(report)
      except BaseException as reporting_error:
        reporting_errors.append(reporting_error)

    # In case reporting failed, generate an unserialized extra report
    for encoding_error in encoding_errors:
      extra_report, _ = self._generate_failure_report(encoding_error,
                                                      phase='encoding',
                                                      serialize=False)
      if extra_report is not None:
        try:
          self._report(extra_report)
        except BaseException as reporting_error:
          reporting_errors.append(reporting_error)

    # If the unserialized report also fails, we must raise
    reporting_errors = list({id(error): error
                             for error in reporting_errors}.values())
    if len(reporting_errors) == 1:
      raise reporting_errors[0]
    if reporting_errors:
      raise BaseExceptionGroup('Failure reporting failed',
                               reporting_errors) from None

    # Return the failure report
    return report

  def forward(self,
              report: FailureReport,
              exception: BaseException | None = None) -> None:
    """Stores and forwards an existing report and registers its Exceptions.

    Does not log again.

    Args:
      report: Existing report, with its original ID and source details.
      exception: Associated Exception or ExceptionGroup, if available.
        Can be provided later by calling forward() again with the same report.
        Otherwise, raise_if_failed() reconstructs it from the report's nodes.
    """

    new = self._store_report(report, exception)

    # Register the Exceptions if provided
    if exception is not None:
      members = tuple(self._iter_exception_members(exception))
      with self._lock:
        for member in members:
          self._members.setdefault(id(member), (member, report.failure_id))

    # Send the report to upstream Managers
    if new:
      self._send(report)

  def snapshot(self) -> tuple[tuple[FailureReport, BaseException | None], ...]:
    """Returns stored reports and their Exceptions in recording order.

    Returns:
      Tuple of (report, exception) pairs. An Exception is None until it has
      been attached to its report. The Exceptions themselves are not copied.
    """

    with self._lock:
      return tuple(self._reports.values())

  def raise_if_failed(self, message: str) -> None:
    """Raises all stored failures together, without logging them again.

    Does nothing when no failures are stored.

    Args:
      message: Message used for the final ExceptionGroup or BaseExceptionGroup.

    Raises:
      ValueError: A local-only report has no attached Exception or node graph.
      FailureTransportError: A received graph has invalid group containment.
      BaseExceptionGroup: The stored failures, including any interruptions.
    """

    failures = self.snapshot()
    # Intermediate owners can retain handle without restoring their objects
    for report, exception in failures:
      if exception is None:
        exception, decoding_errors = self._decode_exception(report)
        self.forward(report, exception)
        # If there were errors while reconstructing, they should be reported
        for decoding_error in decoding_errors:
          try:
            self.capture_exception(decoding_error, phase='decoding')
          except BaseException as reporting_error:
            self._generate_failure_report(reporting_error, phase='reporting',
                                          serialize=False)
    failures = self.snapshot()

    # Generate one ExceptionGroup from all the gathered Exceptions, and raise
    group = self._make_exception_group(
        message, (exception for _, exception in failures
                  if exception is not None))
    if group is not None:
      raise group from None

  def clear(self) -> None:
    """Removes stored reports and releases references to their Exceptions.

    Call only after failure collection has stopped.
    """

    with self._lock:
      self._members.clear()
      self._reports.clear()

  def _record_context_exception(self, error: BaseException, step: str) -> None:
    """Stores and logs an Exception caught inside collect().

    Logging failures are also kept in the current collection.

    Args:
      error: Exception or ExceptionGroup caught by the operation.
      step: Operation name added to Exception notes.
    """

    # Add the caught Exception to the local stack if relevant
    retained = self._add_to_stack(error, step)

    if retained is not None:
      # Store the Exception, generate a report and report the Exception
      try:
        self.capture_exception(retained)
      # Add the reporting failure to the local stack if any
      except BaseException as reporting_error:
        self._add_to_stack(reporting_error, f"report failure from {step}")

  def _add_to_stack(self,
                    error: BaseException,
                    step: str) -> BaseException | None:
    """Adds new failures to the current collect() context.

    Ignores StopRequested and Exceptions already seen in this collection.
    Must be called inside collect().

    Args:
      error: Exception or ExceptionGroup to add.
      step: Operation name added to Exception notes.

    Returns:
      Retained Exception or group, or None if no new failure remains.
    """

    stack = self._collections.stack
    if not stack:
      raise RuntimeError("record() must be used inside collect()")

    # Only store truly unexpected Exceptions
    label, errors, seen = stack[-1]
    error = self._strip_cancellations(error)
    if error is None:
      return None

    # Only add new Exceptions to the encountered Exceptions
    error = self._filter_exception(error,
                                   lambda member: id(member) not in seen)
    if error is None:
      return None

    # Add a generic note if the Exception doesn't already have one
    note = f"{label} step: {step}"
    if note not in getattr(error, '__notes__', ()):
      BaseException.add_note(error, note)

    # Add the error to the list of errors on the stack and mark it and all its
    # children as seen
    errors.append(error)
    seen.update((id(member), member)
                for member in self._iter_exception_members(error))
    return error

  def _generate_failure_report(self,
                               error: BaseException,
                               phase: str | None = None,
                               serialize: bool = True
                               ) -> tuple[FailureReport | None,
                                          tuple[BaseException, ...]]:
    """Creates and stores a report for new failures, without logging it.

    Ignores StopRequested and Exceptions already registered by this manager.

    Args:
      error: Exception or ExceptionGroup to record.
      phase: Phase for this report. Uses the manager's current phase when None.
      serialize: Convert the report to a shareable format.

    Returns:
      New report, or None if no new failure remains, and any interruptions
      encountered during optional serialization.
    """

    # Only handle truly unexpected Exceptions
    error = self._strip_cancellations(error)
    if error is None:
      return None, ()

    # Register the Exception with this manager if it is new
    failure_id = uuid4().hex
    error = self._register_exception(error, failure_id)
    if error is None:
      return None, ()

    # Create a FailureReport for the registered Exception and store it
    try:
      diagnostic = self._format_exception(error)
      nodes, encoding_errors = self._encode_exception(error, serialize)
      report = FailureReport(failure_id=failure_id,
                             block_name=self.block_name,
                             owner_path=self.owner_path,
                             process_name=self.process_name,
                             pid=self.pid,
                             phase=self.phase if phase is None else phase,
                             exception_module=nodes[0].module,
                             exception_type=nodes[0].type_name,
                             exception_message=nodes[0].message,
                             formatted_traceback=diagnostic,
                             nodes=nodes,
                             root=0)
      self._store_report(report, error)
    # If it fails, un-register the Exception and re-raise it
    except BaseException:
      self._unregister_exception(failure_id)
      raise

    return report, encoding_errors

  def _encode_exception(self,
                        error: BaseException,
                        serialize: bool
                        ) -> tuple[tuple[ExceptionNode, ...],
                                   tuple[BaseException, ...]]:
    """Snapshots an Exception graph before attempting optional pickling.

    Args:
      error: Exception or group to describe.
      serialize: Whether to convert data to a shareable format.

    Returns:
      Immutable diagnostic nodes and any interruption from serialization.
    """

    members: list[BaseException] = list()
    indexes: dict[int, int] = dict()
    pending = [error]

    # Start with listing all the Exceptions to handle
    while pending:
      member = pending.pop()
      # Only process a same Exception once
      if id(member) in indexes:
        continue
      indexes[id(member)] = len(members)
      members.append(member)
      related = list(member.exceptions) if isinstance(
          member, BaseExceptionGroup) else list()
      # Also include the causes and context Exceptions
      related.extend(link for link in (member.__cause__, member.__context__)
                     if link is not None)
      pending.extend(reversed(related))

    # Populate the list of Exception nodes
    nodes = list()
    for member in members:
      children = tuple()
      group_message = None
      # ExceptionGroups have extra fields to populate
      if isinstance(member, BaseExceptionGroup):
        children = tuple(indexes[id(child)] for child in member.exceptions)
        group_message = member.message
      # Create all the fields to populate
      notes = tuple(note for note in getattr(member, '__notes__', ())
                    if type(note) is str)
      code = member.code if isinstance(member, SystemExit) else None
      if type(code) not in (int, str, bool, type(None)):
        notes += ('The original SystemExit code is not a simple value.',)
        code = None
      try:
        stack = ''.join(format_tb(member.__traceback__))
      except Exception as stack_error:
        stack = f"<stack unavailable: {type(stack_error).__name__}>"

      # Actually build the Exception node from the generated fields
      nodes.append(ExceptionNode(
          module=type(member).__module__,
          type_name=type(member).__qualname__,
          message=self._safe_message(member),
          mro=tuple(f'{base.__module__}.{base.__qualname__}'
                    for base in type(member).__mro__),
          notes=notes,
          formatted_stack=stack,
          children=children,
          group_message=group_message,
          cause=(indexes[id(member.__cause__)]
                 if member.__cause__ is not None else None),
          context=(indexes[id(member.__context__)]
                   if member.__context__ is not None else None),
          suppress_context=member.__suppress_context__,
          system_exit_code=code))

    if serialize:
      for index, member in enumerate(members):
        # Only serialize the Exceptions, not their containing groups
        if type(member) in (ExceptionGroup, BaseExceptionGroup):
          continue
        try:
          original = member
          # Subclasses of ExceptionGroups are still handled to preserve their
          # extra information
          if isinstance(member, BaseExceptionGroup):
            children = tuple(Exception() if isinstance(child, Exception)
                             else BaseException()
                             for child in member.exceptions)
            original = self._derive_group(member, children)
            if type(original) is not type(member):
              raise TypeError("derive() did not retain the custom group class")
          # Try to pickle the Exception and replace its node information
          payload = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)
          nodes[index] = replace(nodes[index], original_bytes=payload)
        # Include any serialization error in the Exception node
        except Exception as serialization_error:
          reason = (f'{type(serialization_error).__module__}.'
                    f'{type(serialization_error).__qualname__}: '
                    f'{self._safe_message(serialization_error)}')
          nodes[index] = replace(nodes[index], serialization_error=reason)
        except BaseException as interruption:
          reason = (f'Optional serialization interrupted by '
                    f'{type(interruption).__module__}.'
                    f'{type(interruption).__qualname__}')
          return (tuple(replace(node, original_bytes=None,
                                serialization_error=reason) for node in nodes),
                  (interruption,))

    return tuple(nodes), ()

  def _decode_exception(self,
                        report: FailureReport
                        ) -> tuple[BaseException, tuple[BaseException, ...]]:
    """Decodes a serialized Exception report, restoring its original structure.

    Nodes whose reconstruction fails receive a fallback.

    Args:
      report: Received report containing a complete diagnostic graph.

    Returns:
      Reconstructed root Exception or group, and any interruptions from
      reconstruction.

    Raises:
      ValueError: The report contains no encoded graph.
      FailureTransportError: Group containment is cyclic or inconsistent.
    """

    if not report.nodes or report.root is None:
      raise ValueError("Reports must be reconstructed before aggregation")

    # Validate report structure before invoking any reconstruction code
    order = list()
    completed: set[int] = set()
    active: set[int] = set()
    for index in range(len(report.nodes)):
      pending = [(index, False)]
      while pending:
        index, visited = pending.pop()
        # Only handle each Exception once
        if index in completed:
          continue
        node = report.nodes[index]
        if visited:
          active.remove(index)
          completed.add(index)
          order.append(index)
          continue
        # Make sure that the ExceptionGroup structure is valid
        is_group = 'builtins.BaseExceptionGroup' in node.mro
        if (is_group != (node.group_message is not None) or
            is_group != bool(node.children) or index in active):
          raise FailureTransportError('Invalid exception group containment',
                                      failure_id=report.failure_id)
        active.add(index)
        pending.append((index, True))
        pending.extend((child, False) for child in reversed(node.children))

    restored: dict[int, BaseException] = dict()
    with self._lock:
      identities = set(self._members)
    decoding_errors = list()
    # Iterate over all the received Exceptions
    for index in order:
      node = report.nodes[index]
      original = None
      reason = node.serialization_error
      # If the Exception has been serialized, unpickle it
      if node.original_bytes is not None:
        try:
          original = pickle.loads(node.original_bytes)
          if (not isinstance(original, BaseException) or
              type(original).__module__ != node.module or
              type(original).__qualname__ != node.type_name or
              tuple(f'{base.__module__}.{base.__qualname__}'
                    for base in type(original).__mro__) != node.mro):
            raise TypeError("Reconstructed Exception type or MRO changed")
          if id(original) in identities:
            raise TypeError("Distinct nodes reconstructed to one Exception")
        # Reconstruction errors must be reported
        except Exception as reconstruction_error:
          reason = (f'{type(reconstruction_error).__module__}.'
                    f'{type(reconstruction_error).__qualname__}: '
                    f'{self._safe_message(reconstruction_error)}')
          original = None
        except BaseException as interruption:
          decoding_errors.append(interruption)
          reason = (f'Reconstruction interrupted by '
                    f'{type(interruption).__module__}.'
                    f'{type(interruption).__qualname__}')
          original = None

      # Reconstruct an ExceptionGroup
      if node.group_message is not None:
        children = tuple(restored[child] for child in node.children)
        if (original is not None and
            (not isinstance(original, BaseExceptionGroup)
             or original.message != node.group_message)):
          reason = 'Reconstructed group type or message changed'
          original = None
        if isinstance(original, BaseExceptionGroup):
          try:
            original = self._derive_group(original, children)
          except BaseException as interruption:
            decoding_errors.append(interruption)
            reason = (f'Group derivation interrupted by '
                      f'{type(interruption).__module__}.'
                      f'{type(interruption).__qualname__}')
            original = None
          if (original is not None and
              (type(original).__module__ != node.module or
               type(original).__qualname__ != node.type_name)):
            reason = 'derive() did not retain the custom group class'
            original = None
        if original is None:
          original = BaseExceptionGroup(node.group_message, children)
          if (node.module != 'builtins' or
              node.type_name not in ('ExceptionGroup', 'BaseExceptionGroup')):
            reason = reason or 'No usable custom group prototype'

      # Reconstruct a single Exception
      elif original is None:
        original = RemoteFailureMetadata.from_report(report, index, reason)
        reason = reason or 'No original Exception payload'

      # Replace metadata from pickle with the diagnostic snapshot
      original.__traceback__ = None
      original.__notes__ = list(node.notes)
      if reason is not None:
        BaseException.add_note(original,
                               f'Original {node.module}.{node.type_name} '
                               f'could not be restored: {reason}')
      BaseException.add_note(
          original, f'Remote failure {report.failure_id}: Block '
          f'{report.block_name}, owner {report.owner_path}, process '
          f'{report.process_name} (PID {report.pid}), phase {report.phase}')
      if node.formatted_stack:
        BaseException.add_note(
            original, f'Original remote stack:\n{node.formatted_stack}')

      restored[index] = original
      identities.add(id(original))

    # Finalize reconstruction
    for index, original in restored.items():
      node = report.nodes[index]
      original.__cause__ = (restored[node.cause]
                            if node.cause is not None else None)
      original.__context__ = (restored[node.context]
                              if node.context is not None else None)
      original.__suppress_context__ = node.suppress_context

    return restored[report.root], tuple(decoding_errors)

  def _register_exception(self,
                          error: BaseException,
                          failure_id: str) -> BaseException | None:
    """Registers new Exceptions so they are not reported again.

    The check uses Exception objects, not their types or messages.
    A lock prevents two threads from registering the same Exception twice.

    Args:
      error: Exception or ExceptionGroup whose members should be checked.
      failure_id: Report ID assigned to the newly registered Exceptions.

    Returns:
      Newly registered Exception or group, or None if no new individual
      Exception remains.
    """

    members = tuple(self._iter_exception_members(error))

    # Protect with a lock to avoid concurrent Exception registrations
    with self._lock:
      # Handle only members that have never been seen
      fresh = {id(member): member for member in members
               if id(member) not in self._members}
      # If the only new objects are ExceptionGroups, do not remember them
      if not any(not isinstance(member, BaseExceptionGroup)
                 for member in fresh.values()):
        return None
      # Remember all the new Exceptions and groups
      for identity, member in fresh.items():
        self._members[identity] = (member, failure_id)

    # Return the Exceptions that have been registered
    try:
      return self._filter_exception(error,
                                    lambda member: id(member) in fresh)
    # If it fails, un-register the Exception and re-raise it
    except BaseException:
      self._unregister_exception(failure_id)
      raise

  def _unregister_exception(self, failure_id: str) -> None:
    """Removes registrations for one failure if creating its report failed.

    Registrations for other failures are left unchanged.

    Args:
      failure_id: ID of the failure whose registrations should be removed.
    """

    with self._lock:
      self._members = {identity: entry
                       for identity, entry in self._members.items()
                       if entry[1] != failure_id}

  def _store_report(self,
                    report: FailureReport,
                    exception: BaseException | None = None) -> bool:
    """Stores a report without adding the same failure ID twice.

    An Exception can be attached later if it was missing from the first call.

    Args:
      report: FailureReport to store.
      exception: Associated Exception or ExceptionGroup, if available.

    Returns:
      True for a new failure ID. False if the ID was already stored, including
      when a missing Exception was just attached.

    Raises:
      FailureTransportError: The same ID was used for different reports.
    """

    if not isinstance(report, FailureReport):
      raise TypeError("report must be a FailureReport")
    if exception is not None and not isinstance(exception, BaseException):
      raise TypeError("exception must be a BaseException or None")

    # Protect with a lock to avoid concurrent FailureReport registrations
    with self._lock:
      previous = self._reports.get(report.failure_id)
      # Case when a report already exists for this Exception
      if previous is not None:
        previous_report, previous_exception = previous
        # Cannot overwrite an existing report
        if previous_report != report:
          raise FailureTransportError("A failure ID was reused with "
                                      "inconsistent report contents",
                                      failure_id=report.failure_id,
                                      process_name=report.process_name,
                                      pid=report.pid,
                                      phase=report.phase)
        # Case when the existing report wasn't attached to an Exception
        if previous_exception is None and exception is not None:
          self._reports[report.failure_id] = (report, exception)
        return False

      # If there was no previous report, just store the provided one
      self._reports[report.failure_id] = (report, exception)
      return True

  def _record_transport_error(self, error: BaseException) -> None:
    """Records an error when receiving reports from children FailureManagers.

    Args:
      error: Send, receive, or wire-validation error.
    """

    try:
      self.capture_exception(error, phase='transport')
    except BaseException as reporting_error:
      self._generate_failure_report(reporting_error, phase='reporting',
                                    serialize=False)

  def _report(self, report: FailureReport) -> None:
    """Logs and sends a failure report.

    If logging fails, stores that error and tries stderr before raising it.
    If stderr also fails, keeps and raises both reporting errors.

    Args:
      report: FailureReport to log and send.
    """

    reports = [report]
    errors: list[BaseException] = list()

    # Try to report the failure with the logger
    if not self._logger_failed:
      try:
        self._logger.error(str(report))
      # If it fails, create a new FailureReport for this new failure
      except BaseException as error:
        self._logger_failed = True
        BaseException.add_note(error, f"Reporting failure {report.failure_id}")
        infrastructure_report, encoding_errors = self._generate_failure_report(
            error, phase='reporting')
        if infrastructure_report is not None:
          reports.append(infrastructure_report)
        errors.extend((error, *encoding_errors))
        for encoding_error in encoding_errors:
          extra_report, _ = self._generate_failure_report(
              encoding_error, phase='encoding', serialize=False)
          if extra_report is not None:
            reports.append(extra_report)

    # If the logger is down, we have to log directly on stderr
    if self._logger_failed:
      try:
        for pending in reports:
          sys.stderr.write(str(pending))
        sys.stderr.flush()
      # If this also fails, generate a new FailureReport for this new failure
      except BaseException as error:
        infrastructure_report, encoding_errors = self._generate_failure_report(
            error, phase='reporting')
        if infrastructure_report is not None:
          reports.append(infrastructure_report)
        errors.extend((error, *encoding_errors))
        for encoding_error in encoding_errors:
          extra_report, _ = self._generate_failure_report(
              encoding_error, phase='encoding', serialize=False)
          if extra_report is not None:
            reports.append(extra_report)

    # In all cases, send the reports upstream
    for pending in reports:
      try:
        self._send(pending)
      except BaseException as error:
        errors.append(error)
    # If we can't send, raise the gathered errors
    errors = list({id(error): error for error in errors}.values())
    if len(errors) == 1:
      raise errors[0]
    if errors:
      raise BaseExceptionGroup('Failure reporting failed', errors) from None

  def _send(self, report: FailureReport) -> None:
    """Sends a failure report to the upstream FailureManager.

    Args:
      report: Stored report with its original ID and source.
    """

    with self._lock:
      queue = self._queue
    if queue is None:
      return

    # Encode the report to send and send it in the queue
    try:
      payload = pickle.dumps(report.to_wire())
      queue.put(payload)
    # If this fails, remember the queue as broken and record the failure
    except BaseException as error:
      with self._lock:
        if self._queue is queue:
          self._queue = None
      BaseException.add_note(error, f"Sending failure {report.failure_id}")
      try:
        self.capture_exception(error, phase='transport')
      except BaseException as reporting_error:
        raise BaseExceptionGroup('Failure reporting failed',
                                 [error, reporting_error]) from None
      raise

  @staticmethod
  def _iter_exception_members(error: BaseException) -> Iterator[BaseException]:
    """Yields an Exception and its group members, visiting each object once.

    Visits the parent before its children, in group order.
    Does not visit causes or contexts.

    Args:
      error: Exception or ExceptionGroup to visit.
    """

    if not isinstance(error, BaseException):
      raise TypeError("error must be a BaseException")

    pending = [error]
    seen: set[int] = set()
    while pending:
      member = pending.pop()
      # Skip Exceptions already encountered
      if id(member) in seen:
        continue
      seen.add(id(member))
      # Yield the current Exception
      yield member
      # Add its children Exceptions to the list of Exception to handle
      if isinstance(member, BaseExceptionGroup):
        pending.extend(reversed(member.exceptions))

  @staticmethod
  def _derive_group(group: BaseExceptionGroup,
                    children: tuple[BaseException, ...]) -> BaseExceptionGroup:
    """Rebuilds an ExceptionGroup with the supplied children.

    Keeps their order and the original traceback, notes, cause and context.
    Uses a built-in group if the original group cannot derive a valid subgroup.

    Args:
      group: Original ExceptionGroup or BaseExceptionGroup.
      children: Members to include in the new group.

    Returns:
      Group containing the supplied children.
    """

    fallback_reason = None
    # First try to derive the subgroup the regular way, and check for success
    try:
      derived = group.derive(children)
      if (not isinstance(derived, BaseExceptionGroup) or
          derived.message != group.message or
          len(derived.exceptions) != len(children) or
          any(actual is not expected
              for actual, expected in zip(derived.exceptions, children))):
        raise TypeError("derive() changed the requested children or message")

    # If it fails, we have to build the subgroup ourselves
    except Exception as error:
      derived = BaseExceptionGroup(group.message, children)
      fallback_reason = (f"Original group {type(group).__module__}."
                         f"{type(group).__qualname__} could not derive "
                         f"a subgroup ({type(error).__name__})")
    derived.__traceback__ = group.__traceback__
    derived.__cause__ = group.__cause__
    derived.__context__ = group.__context__
    derived.__suppress_context__ = group.__suppress_context__
    if hasattr(group, '__notes__'):
      derived.__notes__ = list(group.__notes__)
    if fallback_reason is not None:
      BaseException.add_note(derived, fallback_reason)
    return derived

  def _filter_exception(self,
                        error: BaseException,
                        keep: Callable[[BaseException], bool]
                        ) -> BaseException | None:
    """Keeps Exceptions for which keep returns True.

    Rejecting a group also rejects its children without visiting them.
    Causes and contexts are left unchanged.

    Args:
      error: Exception or ExceptionGroup to filter.
      keep: Function called for each Exception or group. Return True to keep
        an Exception or check a group's members. Return False to discard it
        and any group members.

    Returns:
      Original error if unchanged, a filtered group if needed, or None if
      nothing remains.
    """

    if not isinstance(error, BaseException):
      raise TypeError("error must be a BaseException")

    # Start with the top Exception and then walk through its children
    pending = [(error, False)]
    filtered: dict[int, BaseException | None] = dict()
    while pending:
      member, visited_children = pending.pop()
      identity = id(member)

      # If the error has already been filtered, no need to filter again
      if identity in filtered:
        continue

      # When first encountering an ExceptionGroup, retrieve all its children
      if not visited_children:
        # If a member doesn't satisfy the criterion, do not consider its
        # children
        if not keep(member):
          filtered[identity] = None
        # If the member is a bare Exception and passes, keep it as filtered
        elif not isinstance(member, BaseExceptionGroup):
          filtered[identity] = member
        # Otherwise mark the ExceptionGroup and its children as pending
        else:
          pending.append((member, True))
          pending.extend((child, False) for child in
                         reversed(member.exceptions))
        continue

      # All bare Exceptions should be handled in the above if statement
      assert isinstance(member, BaseExceptionGroup)
      # List all the children that pass the criterion
      children = tuple(child for original in member.exceptions
                       if (child := filtered[id(original)]) is not None)
      # If none of the children meet the criterion, the parent is rejected
      if not children:
        filtered[identity] = None
      # If all the group passes, it can be saved as-is
      elif (len(children) == len(member.exceptions) and
            all(actual is original
                for actual, original in zip(children, member.exceptions))):
        filtered[identity] = member
      # Otherwise, we need to derive a subgroup of the parent ExceptionGroup
      else:
        filtered[identity] = self._derive_group(member, children)

    return filtered[id(error)]

  def _strip_cancellations(self, error: BaseException) -> BaseException | None:
    """Removes StopRequested Exceptions from an Exception or group.

    Args:
      error: Exception or ExceptionGroup to check.

    Returns:
      Error without StopRequested members, or None if nothing remains.
    """

    return self._filter_exception(error,
                                  lambda member: not isinstance(member,
                                                                StopRequested))

  def _make_exception_group(self,
                            message: str,
                            errors: Iterable[BaseException]
                            ) -> BaseExceptionGroup | None:
    """Groups errors together after removing StopRequested Exceptions.

    Even a single failure is returned in a group.

    Args:
      message: Message used for the new group.
      errors: Exceptions or ExceptionGroups to include.

    Returns:
      ExceptionGroup or BaseExceptionGroup, or None if no failures remain.
    """

    failures = tuple(error for candidate in errors
                     if (error := self._strip_cancellations(candidate))
                     is not None)
    return BaseExceptionGroup(message, failures) if failures else None

  @staticmethod
  def _safe_message(error: BaseException) -> str:
    """Returns the Exception's message, or a short replacement if str() fails.

    Args:
      error: Exception whose message should be read.
    """

    try:
      return str(error)
    except Exception as formatting_error:
      return f"<message unavailable: {type(formatting_error).__name__}>"

  def _format_exception(self, error: BaseException) -> str:
    """Returns the traceback text as Python would display it in a terminal.

    Includes group members, exception chains and notes. Does not include local
    variable values. If formatting fails, lists the Exception types and
    messages instead.

    Args:
      error: Exception or ExceptionGroup to format.
    """

    # Use the convenience function of Python for building the Traceback
    try:
      diagnostic = TracebackException.from_exception(
          error,
          capture_locals=False,
          max_group_width=sys.maxsize,
          max_group_depth=sys.maxsize)
      return ''.join(diagnostic.format(chain=True))

    # If Traceback formatting fails, list the Exception types and messages
    # Use a placeholder if a message cannot be rendered
    except Exception as formatting_error:
      reason = type(formatting_error).__name__
      lines = [f"Traceback formatting failed: {reason}\n"]
      nodes, _ = self._encode_exception(error, serialize=False)
      for index, node in enumerate(nodes):
        lines.append(f"Exception node {index}: "
                     f"{node.module}.{node.type_name}: "
                     f"{node.message}\n{node.formatted_stack}")
        lines.extend(f"{note}\n" for note in node.notes)
        if node.children:
          lines.append(f"Group children: {node.children}\n")
        if node.cause is not None or node.context is not None:
          lines.append(f"Cause: {node.cause}, context: {node.context}, "
                       f"suppress context: {node.suppress_context}\n")
      return ''.join(lines)
