# coding: utf-8

from typing import Optional, Union
from collections.abc import Iterable, Callable
import logging
from re import split
from time import time

from .meta_block import Block


class Pause(Block):
  """This Block parses the data it receives and checks if this data meets the
  given pause criteria. If so, the other Blocks are paused until the criteria
  are no longer met.

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
               criteria: Union[str, Callable, Iterable[Union[str, Callable]]],
               freq: Optional[float] = 50,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      criteria: A :obj:`str`, a :obj:`~collections.abc.Callable`, or an
        :obj:`~collections.abc.Iterable` (like a :obj:`tuple` or a :obj:`list`)
        containing such objects. Each :obj:`str` or
        :obj:`~collections.abc.Callable` represents one pause criterion. There
        is no limit to the given number of stop criteria. If a criterion is
        given as an :obj:`~collections.abc.Callable`, it should accept as its
        sole argument the output of the
        :meth:`crappy.blocks.Block.recv_all_data` method and return :obj:`True`
        if the criterion is met, and :obj:`False` otherwise. If the criterion
        is given as a :obj:`str`, it should follow one the following syntaxes :
        ::

          '<lab> > <threshold>'
          '<lab> < <threshold>'

        With ``<lab>`` and ``<threshold>`` to be replaced respectively with the
        name of a received label and a threshold value. The spaces in the
        string are ignored.
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

    # Handling the case when only one stop condition is given
    if isinstance(criteria, str) or isinstance(criteria, Callable):
      criteria = (criteria,)
    criteria = tuple(criteria)

    self._raw_crit: tuple[Union[str,
                                Callable[[dict[str, list]], bool]]] = criteria
    self._criteria: Optional[tuple[Callable[[dict[str, list]], bool]]] = None

  def prepare(self) -> None:
    """Converts all the given criteria to :obj:`~collections.abc.Callable`."""

    # This operation cannot be performed during __init__ due to limitations of
    # the spawn start method of multiprocessing
    self._criteria = tuple(map(self._parse_criterion, self._raw_crit))

  def loop(self) -> None:
    """Receives data from upstream Blocks, checks if this data meets at least
    one criterion, and puts the other Blocks in pause if that's the case."""

    if not (data := self.recv_all_data()):
      self.log(logging.DEBUG, "No data received during this loop")
      return

    # Pausing only if not paused, and stop criterion is met
    if (self._criteria and any(crit(data) for crit in self._criteria)
        and not self._pause_event.is_set()):
      self.log(logging.WARNING, "Stop criterion reached, pausing the Blocks !")
      self._pause_event.set()
      return

    if (self._criteria and not any(crit(data) for crit in self._criteria)
        and self._pause_event.is_set()):
      self.log(logging.WARNING, "Stop criterion no longer satisfied, "
                                "un-pausing the Blocks !")
      self._pause_event.clear()
      return

    self.log(logging.DEBUG, "No pausing or un-pausing during this loop")

  def _parse_criterion(self,
                       criterion: Union[str, Callable[[dict[str, list]], bool]]
                       ) -> Callable[[dict[str, list]], bool]:
    """Parses a Callable or string criterion given as an input by the user, and
    returns the associated Callable."""

    # If the criterion is already a callable, returning it
    if isinstance(criterion, Callable):
      self.log(logging.DEBUG, "Criterion is a callable")
      return criterion

    # Second case, the criterion is a string containing '<'
    if '<' in criterion:
      self.log(logging.DEBUG, "Criterion is of type var < thresh")
      var, thresh = split(r'\s*<\s*', criterion)

      # Return a function that checks if received data is inferior to threshold
      def cond(data: dict[str, list]) -> bool:
        """Criterion checking that the label values are below a given
        threshold."""

        if var in data:
          return any((val < float(thresh) for val in data[var]))
        return False

      return cond

    # Third case, the criterion is a string containing '>'
    elif '>' in criterion:
      self.log(logging.DEBUG, "Criterion is of type var > thresh")
      var, thresh = split(r'\s*>\s*', criterion)

      # Special case for a time criterion
      if var == 't(s)':
        self.log(logging.DEBUG, "Criterion is about the elapsed time")

        # Return a function that checks if the given time was reached
        def cond(_: dict[str, list]) -> bool:
          """Criterion checking if a given delay is expired."""

          return time() - self.t0 > float(thresh)

        return cond

      # Regular case
      else:

        # Return a function that checks if received data is superior to
        # threshold
        def cond(data: dict[str, list]) -> bool:
          """Criterion checking that the label values are above a given
          threshold."""

          if var in data:
            return any((val > float(thresh) for val in data[var]))
          return False

        return cond

    # Otherwise, it's an invalid syntax
    else:
      raise ValueError("Wrong syntax for the criterion, please refer to the "
                       "documentation")
