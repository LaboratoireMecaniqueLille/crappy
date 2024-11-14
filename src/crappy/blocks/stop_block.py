# coding: utf-8

from typing import Optional, Union
from collections.abc import Iterable, Callable
import logging
from re import split
from time import time

from .meta_block import Block


class StopBlock(Block):
  """This Block parses the data it receives and checks if this data meets the
  given stop criteria. If so, its stops the test.

  Along with the :class:`~crappy.blocks.StopButton` Block, it allows to stop a
  test in a clean way without resorting to CTRL+C.

  .. versionadded:: 2.0.0
  """

  def __init__(self,
               criteria: Union[str, Callable, Iterable[Union[str, Callable]]],
               freq: Optional[float] = 30,
               display_freq: bool = False,
               debug: Optional[bool] = False
               ) -> None:
    """Sets the arguments and initialize the parent class.

    Args:
      criteria: A :obj:`str`, a :obj:`~collections.abc.Callable`, or an
        :obj:`~collections.abc.Iterable` (like a :obj:`tuple` or a :obj:`list`)
        containing such objects. Each :obj:`str` or
        :obj:`~collections.abc.Callable` represents one stop criterion. There
        is no limit to the given number of stop criteria. If a criterion is
        given as an :obj:`~collections.abc.Callable`, it should accept as its
        sole argument the output of the
        :meth:`crappy.blocks.Block.recv_all_data` method and return :obj:`True`
        if the criterion is met, and :obj:`False` otherwise. If the criterion
        is given as a :obj:`str`, it should follow the following syntax :
        ::

          '<lab> > <threshold>'
          '<lab> < <threshold>'

        With ``<lab>`` and ``<threshold>`` to be replaced respectively with the
        name of a received label, and a threshold value. The spaces in the
        string are ignored.
      freq: The target looping frequency for the Block. If :obj:`None`, loops
        as fast as possible.
      display_freq: if :obj:`True`, displays the looping frequency of the
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
    self.pausable = False

    # Handling the case when only one stop condition is given
    if isinstance(criteria, str) or isinstance(criteria, Callable):
      criteria = (criteria,)
    criteria = tuple(criteria)

    self._raw_crit = criteria
    self._criteria = None

  def prepare(self) -> None:
    """Converts all the given criteria to :ref:`collections.abc.Callable`."""

    # This operation cannot be performed during __init__ due to limitations of
    # the spawn start method of multiprocessing
    self._criteria = tuple(map(self._parse_criterion, self._raw_crit))

  def loop(self) -> None:
    """Receives data from upstream Blocks, checks if this data meets the
    criteria, and stop the test if that's the case."""

    data = self.recv_all_data()

    if self._criteria and any(crit(data) for crit in self._criteria):
      self.log(logging.WARNING, "Stop criterion reached, stopping all the "
                                "Blocks !")
      self.stop()

    self.log(logging.DEBUG, "No stop criterion reached during this loop")
  
  def _parse_criterion(self, criterion: Union[str, Callable]) -> Callable:
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
