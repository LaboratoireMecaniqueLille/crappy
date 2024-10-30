# coding: utf-8

from multiprocessing import Pipe
from time import time
from copy import deepcopy
from typing import Union, Any, Optional
from collections.abc import Callable, Iterable
from collections import defaultdict
from select import select
from platform import system
from multiprocessing import current_process
import logging

from .._global import LinkDataError

ModifierType = Callable[[dict[str, Any]], dict[str, Any]]


class Link:
  """This class is used for transferring information between two instances of
  :class:`~crappy.blocks.Block`.

  The created Link is unidirectional, from the input Block to the output Block.
  Under the hood, a Link is basically a :obj:`multiprocessing.Pipe` with
  extra features.

  Note:
    It is possible to add one or multiple :class:`~crappy.modifier.Modifier` to
    modify the transferred value. The Modifiers should be callables taking a
    :obj:`dict` as argument and returning a :obj:`dict`. They can be functions,
    or preferably children of :class:`~crappy.modifier.Modifier`.
  
  .. versionadded:: 1.4.0
  """

  _count = 0

  def __init__(self,
               input_block,
               output_block,
               modifiers: Optional[list[ModifierType]] = None,
               name: Optional[str] = None) -> None:
    """Sets the instance attributes.

    Args:
      input_block: The Block sending data through the Link.
      output_block: The Block receiving data through the Link.
      modifiers: A :obj:`list` containing callables. If several objects given,
        they will be called in the given order. Refer to
        :class:`~crappy.modifier.Modifier` for more information.
      name: Name of the Link, to differentiate it from the others when
        debugging. If no specific name is given, the Links are numbered in the
        order in which they are instantiated in the script.
    
    .. versionchanged:: 1.5.9 renamed *condition* argument to *conditions*
    .. versionchanged:: 1.5.9 renamed *modifier* argument to *modifiers*
    .. versionremoved:: 2.0.0 *conditions*, *timeout* and *action* arguments
    """

    if modifiers is None:
      modifiers = list()

    # Checking that all the given modifiers are callable
    if modifiers and not all(callable(mod) for mod in modifiers):
      not_callable = [mod for mod in modifiers if not callable(mod)]
      raise TypeError(f"The following objects passed as modifiers are not "
                      f"callable : {not_callable} !")

    self.name = name if name is not None else f'link{self._get_count()}'
    self._in, self._out = Pipe()
    self._modifiers = modifiers

    # Associating the link to the input and output blocks
    input_block.add_output(self)
    output_block.add_input(self)

    self._last_warn = time()
    self._logger: Optional[logging.Logger] = None
    self._system = system()

  def __new__(cls, *args, **kwargs):
    """When instantiating a new Link, increments the Link counter."""

    cls._count += 1
    return super().__new__(cls)

  @classmethod
  def _get_count(cls) -> int:
    """Returns the current number of instantiates Links, as an :obj:`int`."""

    return cls._count

  def log(self, log_level: int, msg: str) -> None:
    """Method for recording log messages from the Link.

    Args:
      log_level: An :obj:`int` indicating the logging level of the message.
      msg: The message to log, as a :obj:`str`.
    
    .. versionadded:: 2.0.0
    """

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{self.name}")

    self._logger.log(log_level, msg)

  def poll(self) -> bool:
    """Returns :obj:`True` if there's data available for reading.
    
    .. versionadded:: 2.0.0
    """

    return self._in.poll()

  def send(self, value: dict[str, Any]) -> None:
    """Sends a value from the upstream Block to the downstream Block.

    Before sending, applies the given Modifiers and makes sure there's room in
    the Pipe for sending the data (Linux only).
    """

    # Applying the modifiers to the value to send
    for mod in self._modifiers:
      value = mod(deepcopy(value))
      # No need to continue if there's no value to send anymore
      if value is None:
        return

    if not isinstance(value, dict):
      self.log(logging.ERROR, f"Trying to send object of type {type(value)} "
                              f"instead of dict !")
      raise LinkDataError

    # Finally, sending the dict to the link
    if self._system == 'Linux':
      # Can only check on Linux if a pipe is full
      if select([], [self._out], [], 0)[1]:
        self._out.send(value)
      # Warning in case the pipe is full
      elif time() - self._last_warn > 1:
          self._last_warn = time()
          self.log(logging.WARNING, f"Cannot send the values, the Link is "
                                    f"full !")
    else:
      self._out.send(value)

  def recv(self) -> dict[str, Any]:
    """Reads a single value from the Link and returns it.

    The read value is the oldest available in the Link, see :meth:`recv_last`
    for reading the newest available value.

    If no data is available in the Link, returns an empty :obj:`dict`.

    Returns:
      A :obj:`dict` whose keys are the labels being sent, and for each key a
      single value (usually a :obj:`float` or a :obj:`str`).
      
    .. versionremoved:: 2.0.0 *blocking* argument
    """

    if self._in.poll():
      return self._in.recv()
    else:
      return dict()

  def recv_last(self) -> dict[str, Any]:
    """Reads all the available values in the Link, and returns the newest one.

    If no data is available in the Link, returns an empty :obj:`dict`. All the
    data that is not returned is permanently dropped.

    Returns:
      A :obj:`dict` whose keys are the labels being sent, and for each key a
      single value (usually a :obj:`float` or a :obj:`str`).
    
    .. versionremoved:: 2.0.0 *blocking* argument
    """

    data = dict()

    while self._in.poll():
      data = self._in.recv()

    return data

  def recv_chunk(self) -> dict[str, list[Any]]:
    """Reads all the available values in the Link, and returns them all.

    Returns:
      A :obj:`dict` whose keys are the labels being sent, and for each key a
      :obj:`list` of the received values. The first item in the list is the
      oldest one available in the Link, the last item is the newest available.
    
    .. versionremoved:: 1.5.9 *length* argument
    .. versionadded:: 1.5.9 *blocking* argument
    .. versionremoved:: 2.0.0 *blocking* argument
    """

    ret = defaultdict(list)

    while self._in.poll():
      data = self._in.recv()
      for label, value in data.items():
        ret[label].append(value)

    return dict(ret)


def link(in_block,
         out_block,
         /, *,
         modifier: Optional[Union[Iterable[ModifierType],
                                  ModifierType]] = None,
         name: Optional[str] = None) -> None:
  """Function linking two Blocks, allowing to send data from one to the other.

  It instantiates a :class:`~crappy.links.Link` between two children of
  :class:`~crappy.blocks.Block`.

  The created Link is unidirectional, from the input Block to the output Block.
  Under the hood, a Link is basically a :obj:`multiprocessing.Pipe` with
  extra features.

  Args:
    in_block: The Block sending data through the Link.

      .. versionchanged:: 2.0.7
         now a positional-only argument
    out_block: The Block receiving data through the Link.

      .. versionchanged:: 2.0.7
         now a positional-only argument
    modifier: Either a callable, or an iterable (like a :obj:`list` or a
      :obj:`tuple`) containing callables. If several given (in an iterable),
      they are called in the given order. They should preferably be children of
      :class:`~crappy.modifier.Modifier`. Refer to  the associated
      documentation for more information.

      .. versionchanged:: 2.0.7
         now a keyword-only argument
    name: Name of the Link, to differentiate it from the others when debugging.
      If no specific name is given, the Links are numbered in the order in
      which they are instantiated in the script.

      .. versionchanged:: 2.0.7
         now a keyword-only argument
      
  .. versionadded:: 1.4.0
  .. versionchanged:: 1.5.9
     explicitly listing the *condition*, *modifier*, *timeout*, *action* and
     *name* arguments
  .. versionremoved:: 2.0.0 *condition*, *timeout* and *action* arguments
  """

  # Forcing the modifiers into lists
  if modifier is not None:
    try:
      iter(modifier)
      modifier = list(modifier)
    except TypeError:
      modifier = [modifier]

  # Actually creating the Link object
  Link(input_block=in_block,
       output_block=out_block,
       modifiers=modifier,
       name=name)
