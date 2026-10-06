# coding: utf-8

from collections.abc import Sequence
from collections import Counter
from typing import Any
from time import monotonic
import logging
from numbers import Real

from .meta_block import Block
from .schedulers import State
from .._global import SchedulerStop


class Scheduler(Block):
  """This Block generates signals according to a user-defined state machine.

  Use it to coordinate a procedure with several phases, such as applying a
  command only after a given event, waiting for a measurement, adjusting a
  trigger based on a feedback, etc. Each phase can produce several related
  output values and choose its next phase based on elapsed time or data
  received from upstream Blocks. Transitions can branch to different States or
  return to an earlier State, allowing feedback-driven workflows and repeated
  phases.

  This Block is an extension of the :class:`~crappy.blocks.Generator` Block,
  which is simpler when the goal is to drive a waveform or setpoint over time.
  Both Blocks can use incoming data in their transition decisions, but a
  Generator's Path order is fixed. Choose the Scheduler when a procedure needs
  multiple coordinated output labels, different possible next States, or custom
  output logic for each State.

  The state machine is built from a sequence of
  :class:`~crappy.blocks.schedulers.State` objects. Each
  State defines functions that generate output values and conditions that
  trigger transitions to other States. Both types of callable receive the
  time elapsed since the current State was entered and the data received from
  upstream Blocks. Stop conditions are evaluated before output functions on
  every loop.

  Output functions may generate only a subset of the requested output labels.
  The Scheduler retains the latest value for every label across loops and
  State transitions, and starts sending only once every output label has a
  value. It then sends whenever a value changes, whenever the State changes,
  or on every loop if ``spam`` is :obj:`True`. Every sent dictionary also
  contains the identifier of the current State.

  The first provided State is the initial one. The identifiers ``'Start'``
  and ``'End'`` are reserved for internal States. A transition to ``'End'``
  completes the workflow and, by default, stops the script after ``end_delay``
  seconds.

  .. versionadded:: 2.1.0
  """

  def __init__(self,
               states: Sequence[State],
               output_labels: Sequence[str],
               input_labels: Sequence[str] | None = None,
               init_values: dict[str, Any] | None = None,
               last_output: dict[str, Any] | None = None,
               spam: bool = False,
               safe_start: bool = False,
               state_id_label: str = 'state',
               end_delay: float | None = 1.0,
               freq: float | None = 200.0,
               display_freq: bool = False,
               debug: bool | None = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      states: A non-empty sequence of :class:`~crappy.blocks.schedulers.State`
        objects defining the state machine. The first State in the sequence is
        used as the initial State. State identifiers must be unique, and every
        transition target must point to a provided State or one of the internal
        ``'Start'`` and ``'End'`` States.
      output_labels: A non-empty sequence of labels that may be sent to
        downstream Blocks. Values returned for other labels are ignored. The
        Scheduler waits until a value is known for every output label before
        sending data. Mandatory, to enforce consistency of the output signal of
        this Block. If some labels are not managed by the first State, you can
        specify their value in ``init_values``.
      input_labels: The labels that will be used by the States to generate
        output values or make transition decisions. When provided, only data
        associated to these labels is passed to the States. If :obj:`None` or
        empty, values are retained for every received label. When
        ``safe_start`` is :obj:`True`, every explicitly listed input label must
        be received before output functions are evaluated. It is strongly
        recommended to set this argument, to enforce consistency of the overall
        State graphs and labels.
      init_values: Initial values for any of the ``output_labels``. They seed
        the output cache before the first State is entered. If they provide
        every output label and ``safe_start`` is :obj:`False`, they are sent
        once from the internal ``'Start'`` State.
      last_output: Values to generate once after entering the internal
        ``'End'`` State. They are merged with the latest output values, so
        labels omitted here retain their previous values. The values provided
        here do not bypass the ``output_labels`` guard, if not all values are
        known for the output labels in the ``'End'`` State then the last output
        won't be sent.
      spam: If :obj:`True`, sends output values on every loop once all
        ``output_labels`` have a value. Otherwise, sends only when a value
        changes or when the current State changes.
      safe_start: If :obj:`True`, does not evaluate State output functions
        until at least one value has been received for every label explicitly
        given in ``input_labels``. This does not delay the evaluation of stop
        conditions.
      state_id_label: The additional output label carrying the identifier of
        the current State. It must not also appear in ``output_labels``.
      end_delay: The delay, in seconds from entering the internal ``'End'``
        State, before stopping the script. If :obj:`None`, the Scheduler stays
        idle in the ``'End'`` State until another Block stops the script.
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
    """

    super().__init__()

    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    # Checking the validity of the provided arguments
    match states:
      case ():
        raise ValueError("The provided sequence of States is empty")
      case (*states,) if all(isinstance(state, State) for state in states):
        # Reject duplicate State ids
        ids = [state.id for state in states]
        if len(states) != len(set(ids)):
          duplicate = [id_ for id_, count in Counter(ids).items() if count > 1]
          raise ValueError(f"Found duplicate State id(s): "
                           f"{', '.join(duplicate)}")
        # Reject States whose id is 'Start' or 'End'
        if any(state.id in ('Start', 'End') for state in states):
          raise ValueError("The State ids 'Start' and 'End' are reserved, "
                           "please choose a different id")
        self._states: dict[str, State] = {state.id: state for state in states}
      case (*_,):
        raise TypeError("All the States must be provided as State instances")
      case _:
        raise TypeError("At least one State must be provided, in a sequence "
                        "of States")

    match output_labels:
      case ():
        raise ValueError("No output_labels were provided! A Scheduler Block "
                         "that does not output data is most certainly useless")
      case (*labels,) if (all(isinstance(label, str) for label in labels) and
                          all(label.strip() for label in labels)):
        self._output_labels: Sequence[str] = tuple(output_labels)
      case (*labels, ) if all(isinstance(label, str) for label in labels):
        raise ValueError("All the output labels must be provided as non-empty "
                         "strings")
      case (*_,):
        raise TypeError("All the output labels must be provided as non-empty "
                        "strings")
      case _:
        raise TypeError("At least one output label must be provided, in a "
                        "sequence of strings")

    match input_labels:
      case None:
       self._input_labels: Sequence[str] = tuple()
      case ():
        self._input_labels: Sequence[str] = tuple()
      case (*labels,) if (all(isinstance(label, str) for label in labels) and
                          all(label.strip() for label in labels)):
        self._input_labels: Sequence[str] = tuple(input_labels)
      case (*labels, ) if all(isinstance(label, str) for label in labels):
        raise ValueError("All the input labels must be provided as non-empty "
                         "strings")
      case (*_,):
        raise TypeError("All the input labels must be provided as non-empty "
                        "strings")
      case _:
        raise TypeError("When provided, input labels should be in a sequence "
                        "of non-empty strings")

    match init_values:
      case None:
        self._init_values: dict[str, Any] = dict()
      case dict(init) if (all(isinstance(key, str) for key in init) and
                          all(key.strip() for key in init) and
                          all(key in self._output_labels for key in init)):
        self._init_values: dict[str, Any] = init_values.copy()
      case dict(init) if (all(isinstance(key, str) for key in init) and
                          all(key.strip() for key in init)):
        missing = [key for key in init_values
                   if key not in self._output_labels]
        raise ValueError(f"An init value is provided for label(s) "
                         f"{', '.join(missing)} although these label(s) are "
                         f"not registered as output label(s)")
      case dict(init) if all(isinstance(key, str) for key in init):
        raise ValueError("The init values must be provided as a dictionary "
                         "whose keys are labels, as non-empty strings")
      case dict():
        raise TypeError("The init values must be provided as a dictionary "
                        "whose keys are labels, as non-empty strings")
      case _:
        raise TypeError("The init values must be provided as a dictionary "
                        "whose keys are labels, as non-empty strings")

    match last_output:
      case None:
        self._last_output: dict[str, Any] = dict()
      case dict(last) if (all(isinstance(key, str) for key in last) and
                          all(key.strip() for key in last) and
                          all(key in self._output_labels for key in last)):
        self._last_output: dict[str, Any] = last_output.copy()
      case dict(last) if (all(isinstance(key, str) for key in last) and
                          all(key.strip() for key in last)):
        missing = [key for key in last_output
                   if key not in self._output_labels]
        raise ValueError(f"An last output is provided for label(s) "
                         f"{', '.join(missing)} although these label(s) are "
                         f"not registered as output label(s)")
      case dict(last) if all(isinstance(key, str) for key in last):
        raise ValueError("The last outputs must be provided as a dictionary "
                         "whose keys are labels, as non-empty strings")
      case dict():
        raise TypeError("The last outputs must be provided as a dictionary "
                        "whose keys are labels, as non-empty strings")
      case _:
        raise TypeError("The last outputs must be provided as a dictionary "
                        "whose keys are labels, as non-empty strings")

    match spam:
      case bool():
        self._spam: bool = spam
      case _:
        raise TypeError("spam mut be provided as a boolean")

    match safe_start:
      case bool():
        self._safe_start: bool = safe_start
      case _:
        raise TypeError("safe_start mut be provided as a boolean")

    match state_id_label:
      case str() if state_id_label.strip():
        # The state id label must not be already present in the output labels
        if state_id_label in self._output_labels:
          raise ValueError("The state id label is already part of the output "
                           "labels")
        self._state_id_label: str = state_id_label
      case str():
        raise ValueError("the state id label must be provided as a non-empty "
                         "string")
      case _:
        raise TypeError("the state id label must be provided as a non-empty "
                        "string")

    match end_delay:
      case None:
        self._end_delay: float | None = None
      case Real() if end_delay >= 0:
        self._end_delay: float | None = float(end_delay)
      case Real():
        raise ValueError("The end delay must be provided as a positive float "
                         "or None")
      case _:
        raise TypeError("The end delay must be provided as a positive float "
                        "or None")

    # Check that the state transitions are consistent
    all_ids = {'Start', 'End', *[state.id for state in self._states.values()]}
    all_targets = {target for state in self._states.values()
                   for _, target in state.stop_conditions}
    if unexpected := all_targets - all_ids:
      raise IOError(f"The State graph contains state transition targets that "
                    f"do not correspond to registered States ids: "
                    f"{', '.join(unexpected)}")

    # Beginning with a special start state for consistency
    self._current_state: State = State('Start', list(),
                                       ((self._true, states[0].id),))
    self._states['Start'] = self._current_state
    # Already defin the end state
    end_state = State('End', (self._last_outputs,),
                      ((self._end_condition, 'End'),))
    self._states['End'] = end_state

    # Other attributes used in the Scheduler workflow
    self._last_t_sched: float = monotonic()
    self._receive_cache: dict[str, Any] = dict()
    self._send_cache: dict[str, Any] = dict()
    self._latest_sent: dict[str, Any] = dict()
    self._send_on_state_change: bool = True
    self._end: bool = False
    self._last_out_sent: bool = False
    self._last_warn_start: float = float('-inf')
    self._last_warn_send: float = float('-inf')
    self._warned_unexpected: set[tuple[str, frozenset[str]]] = set()
    self._warned_overlap: set[tuple[str, frozenset[str]]] = set()

  def begin(self) -> None:
    """Initializes the runtime caches and enters the first provided State.

    If all output values are supplied by ``init_values`` and ``safe_start`` is
    :obj:`False`, also sends these values from the internal ``'Start'`` State.
    """

    self._last_t_sched = monotonic()
    self._send_cache.update(self._init_values)

    # Manage early send if all the necessary values are already present
    if not self._safe_start and all(label in self._send_cache
                                    for label in self._output_labels):
      self.log(logging.INFO, "Sending early output data as all initial values "
                             "for output labels were provided and safe_start "
                             "is False")
      self._send_values(self._send_cache)

    # Will always switch to the first State and set send_on_state_change
    dt = monotonic() - self._last_t_sched
    self._evaluate_conditions(dt, dict())

  def loop(self) -> None:
    """Receives input data and runs one iteration of the state machine.

    The current State's stop conditions are evaluated first. If none causes a
    transition, the State's output functions are evaluated and their values
    are added to the output cache. The cache is sent once every requested
    output label is populated and the configured sending criteria are met.
    """

    # Receive data from upstream Blocks
    data = self.recv_all_data()
    # Update the latest-known values cache with the newly received data
    for label, values in data.items():
      # Store only the input labels to track, if provided
      if not self._input_labels or label in self._input_labels:
        self._receive_cache[label] = values[-1]
    # Filter data labels if the input labels are explicitly provided
    data = {label: values for label, values in data.items()
            if not self._input_labels or label in self._input_labels}
    # Complement the received values with the latest-known ones if necessary
    for label, value in self._receive_cache.items():
      if label not in data:
        data[label] = [value]

    # Time elapsed since the current ste started running
    dt = monotonic() - self._last_t_sched

    # Evaluate the stop conditions and switch to the next state if one is met
    if self._evaluate_conditions(dt, data):
      return

    # Don't evaluate output at all in the End State
    if self._end:
      self.log(logging.DEBUG, "Stopping loop early in End state")
      return

    # Stop early if some input values are missing for a safe start
    if (self._safe_start and not
        all(label in self._receive_cache for label in self._input_labels) and
        self._current_state is not self._states['End']):
      missing = [label for label in self._input_labels
                 if label not in self._receive_cache]
      if monotonic() - self._last_warn_start > 2:
        self.log(logging.WARNING, f"Not evaluating the outputs as safe_start "
                                  f"is True an input values are missing for "
                                  f"label(s) {', '.join(missing)}")
        self._last_warn_start = monotonic()
      return

    # Evaluate the outputs to send
    global_out = self._evaluate_output(dt, data)

    # Checking if there are unexpected labels in the generated outputs
    unexpected = [label for label in global_out
                  if label not in self._output_labels]
    key = self._current_state.id, frozenset(unexpected)
    if unexpected:
      if key not in self._warned_unexpected:
        self.log(logging.WARNING,
                 f"The outputs of state {self._current_state.id} generated "
                 f"data for label(s) {', '.join(unexpected)}, which is/are "
                 f"not in the provided output_labels, ignoring it\nEither "
                 f"update your outputs or the output_labels argument")
        self._warned_unexpected.add(key)
      global_out = {label: value for label, value in global_out.items()
                    if label in self._output_labels}

    # Update the output buffer with the new generated values
    self._send_cache.update(global_out)

    # All the requested out labels must be populated before sending anything
    if not all(label in self._send_cache for label in self._output_labels):
      missing = [label for label in self._output_labels
                 if label not in self._send_cache]
      if monotonic() - self._last_warn_send > 2:
        self.log(logging.WARNING, f"Not sending data to downstream Blocks, "
                                  f"missing values for label(s) "
                                  f"{', '.join(missing)}")
        self._last_warn_send = monotonic()
      return

    # Actually sending the data if spam is True or if it is new
    self._send_values(self._send_cache)

  def finish(self) -> None:
    """Resets the Scheduler's runtime state so it can be started again."""

    self._last_t_sched: float = monotonic()
    self._receive_cache: dict[str, Any] = dict()
    self._send_cache: dict[str, Any] = dict()
    self._latest_sent: dict[str, Any] = dict()
    self._send_on_state_change: bool = True
    self._end: bool = False
    self._last_out_sent: bool = False
    self._last_warn_start: float = float('-inf')
    self._last_warn_send: float = float('-inf')
    self._warned_unexpected: set[tuple[str, frozenset[str]]] = set()
    self._warned_overlap: set[tuple[str, frozenset[str]]] = set()

    # Must be reset here for a potential next run
    self._current_state = self._states['Start']

  def _evaluate_output(self,
                       dt: float,
                       data:  dict[str, list[Any]]) -> dict[str, Any]:
    """Evaluates and combines all the current State's output functions.

    Args:
      dt: The time elapsed since the current State was entered.
      data: The values received from upstream Blocks, grouped by label.

    Returns:
      A dictionary containing the generated output values. If several output
      functions return the same label, the value returned by the last one is
      kept.
    """

    global_out = dict()
    for i, output in enumerate(self._current_state.outputs):
      out = output(dt, data)
      match out:
        case None:
          continue
        case dict() if (all(isinstance(key, str) for key in out) and
                        all(key.strip() for key in out) and
                        not any(label in global_out for label in out)):
          global_out.update(out)
        case dict() if (all(isinstance(key, str) for key in out) and
                        all(key.strip() for key in out)):
          overlap = [label for label in out if label in global_out]
          key = self._current_state.id, frozenset(overlap)
          if key not in self._warned_overlap:
            self.log(logging.WARNING, f"Label(s) {', '.join(overlap)} already "
                                      f"set for output to downstream Blocks, "
                                      f"output {i} of state "
                                      f"{self._current_state.id} will "
                                      f"overwrite")
            self._warned_overlap.add(key)
          global_out.update(out)
        case dict():
          raise ValueError("The dicts returned by State outputs must have "
                           "only non-empty strings as keys")
        case _:
          raise TypeError("State outputs are expected to return either a dict "
                          "or None")

    return global_out

  def _evaluate_conditions(self,
                           dt: float,
                           data:  dict[str, list[Any]]) -> bool:
    """Evaluates the current State's stop conditions in order.

    Args:
      dt: The time elapsed since the current State was entered.
      data: The values received from upstream Blocks, grouped by label.

    Returns:
      :obj:`True` if a condition caused a transition, otherwise :obj:`False`.
    """

    for cond, id_ in self._current_state.stop_conditions:
      if cond(dt, data):
        self._current_state = self._states[id_]
        self.log(logging.INFO, f"Scheduler switched to the "
                               f"{self._current_state.id} State")

        # Reset the Conditions and Outputs if supported
        for output in self._current_state.outputs:
          if hasattr(output, 'reset') and callable(output.reset):
            output.reset()
        for condition, _ in self._current_state.stop_conditions:
          if hasattr(condition, 'reset') and callable(condition.reset):
            condition.reset()

        self._last_t_sched = monotonic()
        self._send_on_state_change = True
        return True
    return False

  def _send_values(self, data: dict[str, Any]) -> None:
    """Sends output values when required by the current sending policy.

    The current State identifier is added under ``state_id_label`` immediately
    before sending.

    Args:
      data: The complete mapping of output labels to their current values.
    """

    if data != self._latest_sent or self._spam or self._send_on_state_change:
      self._latest_sent = data.copy()
      # Add the current state id to the data to send
      to_send = data.copy()
      to_send[self._state_id_label] = self._current_state.id
      self.send(to_send)
      # Lower the send_on_sate_change flag if data was actually sent
      self._send_on_state_change = False
    else:
      self.log(logging.DEBUG, "Not sending data at this loop")

  @staticmethod
  def _true(*_, **__) -> bool:
    """Unconditionally returns :obj:`True` for leaving the ``'Start'``
    State."""

    return True

  def _end_condition(self,
                     dt: float,
                     _:  dict[str, list[Any]]) -> bool:
    """Raises :exc:`~crappy._global.SchedulerStop` after ``end_delay``.

    Delay evaluation is skipped until a configured ``last_output`` has been
    generated, so that these values can be sent before the Scheduler stops.

    Args:
      dt: The time elapsed since the ``'End'`` State was entered.

    Returns:
      Always :obj:`False`. Reaching the configured delay raises an exception
      instead of causing another State transition.
    """

    # Looping forever if there's no en delay
    if self._end_delay is None:
      return False

    # Skip this evaluation regardless of end_delay to send the last output
    if self._last_output and not self._end:
      return False

    # Otherwise wait for the end delay to be exhausted
    if dt > self._end_delay:
      raise SchedulerStop
    return False

  def _last_outputs(self,
                    _: float,
                    __:  dict[str, list[Any]]) -> dict[str, Any] | None:
    """Returns the configured final output values on the first call, if any,
    otherwise :obj:`None`."""

    # Setting the End flag here as this method will always be evaluated
    self._end = True

    if self._last_output and not self._last_out_sent:
      self._last_out_sent = True
      return self._last_output
    return None
