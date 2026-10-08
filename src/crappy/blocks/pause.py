# coding: utf-8

from collections.abc import Sequence, Callable
import logging
import re
from time import time

from .meta_block import Block


class Pause(Block):
  """This Block parses the data it receives and checks if this data meets the
  given pause criteria. If so, the other Blocks are paused until the criteria
  are no longer met.

  Each criterion is evaluated once per loop and initially counts as not met.
  Insufficient data retains its previous state. The Blocks remain paused while
  any criterion's cached state is met.

  When paused, the other Blocks are still looping but no longer executing any
  code. This feature is mostly useful when human intervention on a test setup
  is required, to ensure that nothing happens during that time.

  It is possible to prevent a Block from being affected by a pause by setting
  its ``pausable`` attribute to :obj:`False`. In particular, the Block(s)
  responsible for outputting the labels checked by the criteria should keep
  running, otherwise the test will be put on hold forever.

  Important:
    This Block prevents other Blocks from running normally, but no specific
    mechanism for putting hardware in an idle state is implemented. For
    example, a motor driven by an :class:`~crappy.blocks.Machine` Block might
    keep moving according to the last command it received before the Blocks
    were paused. It is up to the user to put hardware in the desired state
    before starting a pause.

  Warning:
    Using this Block is potentially dangerous, as it leaves hardware
    unsupervised with no software control on it. It is advised to always
    include hardware securities on your setup.

  .. versionadded:: 2.0.7
  """

  def __init__(self,
               criteria: str | Callable | Sequence[str | Callable],
               freq: float | None = 50,
               display_freq: bool = False,
               debug: bool | None = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      criteria: A :obj:`str`, a :obj:`~collections.abc.Callable`, or an
        :obj:`~collections.abc.Iterable` (like a :obj:`tuple` or a :obj:`list`)
        containing such objects. Each :obj:`str` or
        :obj:`~collections.abc.Callable` represents one pause criterion. There
        is no limit to the given number of pause criteria. If a criterion is
        given as an :obj:`~collections.abc.Callable`, it should accept as its
        sole argument the output of the
        :meth:`crappy.blocks.meta_block.block.Block.recv_all_data` method and
        return :obj:`True` if the criterion is met, :obj:`False` if evaluated
        and not met, or :obj:`None` for insufficient data, retaining its
        previous state.
        If the criterion is given as a :obj:`str`, it should follow one the
        following syntaxes:
        ::

          '<lab> > <threshold>'
          '<lab> < <threshold>'

        With ``<lab>`` and ``<threshold>`` to be replaced respectively with the
        name of a received label and a threshold value. The spaces in the
        string are ignored. A string criterion is met if any received value
        satisfies the comparison, absent labels or empty value lists retain
        its previous state. ``'t(s) > <threshold>'`` compares elapsed time on
        every loop, even without incoming data.
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

    self.pausable = False
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    # Handling the case when only one pause condition is given
    if isinstance(criteria, str) or isinstance(criteria, Callable):
      criteria = (criteria,)
    criteria = tuple(criteria)

    self._raw_crit: tuple[str |
                          Callable[[dict[str, list]], bool | None],
                          ...] = criteria
    self._criteria: tuple[Callable[[dict[str, list]], bool | None],
                          ...] | None = None
    self._criterion_states: list[bool] = list()

  def prepare(self) -> None:
    """Converts criteria to callables and resets their cached states to
    :obj:`False`."""

    if not self.inputs:
      raise IOError("No Link pointing towards the Pause Block !")

    # This operation cannot be performed during __init__ due to limitations of
    # the spawn start method of multiprocessing
    self._criteria = tuple(map(self._parse_criterion, self._raw_crit))
    self._criterion_states = [False for _ in self._criteria]

  def loop(self) -> None:
    """Evaluates every criterion once and pauses while any cached state is
    :obj:`True`.

    :obj:`None` results retain their previous state. Time criteria are
    evaluated regardless of the incoming data.
    """

    if not (data := self.recv_all_data()):
      self.log(logging.DEBUG, "No data received during this loop")

    # Evaluate all the available criteria
    for index, criterion in enumerate(self._criteria or tuple()):
      result = criterion(data)
      if result is not None:
        self._criterion_states[index] = bool(result)

    should_pause = any(self._criterion_states)

    # Pausing only if not paused, and a pause criterion is met
    if should_pause and not self._pause_event.is_set():
      self.log(logging.WARNING, "Pause criterion reached, pausing the Blocks!")
      self._pause_event.set()
      return

    if self._criteria and not should_pause and self._pause_event.is_set():
      self.log(logging.WARNING, "Pause criterion no longer satisfied, "
                                "un-pausing the Blocks!")
      self._pause_event.clear()
      return

    self.log(logging.DEBUG, "No pausing or un-pausing during this loop")

  def _parse_criterion(self,
                       criterion: str | Callable[[dict[str, list]],
                                                 bool | None]
                       ) -> Callable[[dict[str, list]], bool | None]:
    """Returns a criterion callable."""

    # If the criterion is already a callable, returning it
    if isinstance(criterion, Callable):
      self.log(logging.DEBUG, "Criterion is a callable")
      return criterion

    # Second case, the criterion is a string containing '<'
    if '<' in criterion:
      self.log(logging.DEBUG, "Criterion is of type var < thresh")
      var, thresh = re.fullmatch(r'\s*(.+?)\s*<\s*(.+?)\s*',
                                 criterion).groups()

      # Return a function that checks if received data is inferior to threshold
      def cond(data: dict[str, list]) -> bool | None:
        """Checks for values below the threshold, or None without samples."""

        if (val := data.get(var)) is not None and val:
          return any((val < float(thresh) for val in val))
        return None

      return cond

    # Third case, the criterion is a string containing '>'
    elif '>' in criterion:
      self.log(logging.DEBUG, "Criterion is of type var > thresh")
      var, thresh = re.fullmatch(r'\s*(.+?)\s*>\s*(.+?)\s*',
                                 criterion).groups()

      # Special case for a time criterion
      if var == 't(s)':
        self.log(logging.DEBUG, "Criterion is about the elapsed time")

        # Return a function that checks if the given time was reached
        def cond(_: dict[str, list]) -> bool:
          """Checks elapsed time independently of received data."""

          return time() - self.t0 > float(thresh)

        return cond

      # Regular case
      else:

        # Return a function that checks if received data is superior to
        # threshold
        def cond(data: dict[str, list]) -> bool | None:
          """Checks for values above the threshold, or None without samples."""

          if (val := data.get(var)) is not None and val:
            return any((val > float(thresh) for val in val))
          return None

        return cond

    # Otherwise, it's an invalid syntax
    else:
      raise ValueError("Wrong syntax for the criterion, please refer to the "
                       "documentation")
