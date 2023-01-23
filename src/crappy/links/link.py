# coding: utf-8

from multiprocessing import Pipe
from time import time
from copy import deepcopy
from typing import Callable, Union, Any, Dict, Optional, List
from collections import defaultdict
from select import select
from multiprocessing import current_process
import logging

from ..modifier import Modifier
from .._global import LinkDataError

ModifierType = Callable[[Dict[str, Any]], Dict[str, Any]]


class Link:
  """This class is used for transferring information between the Blocks.

  The created link is unidirectional, from the input block to the output block.
  Under the hood, a link is basically a :class:`multiprocessing.Pipe` with
  extra features.

  Note:
    It is possible to add one or multiple :ref:`Modifiers` to modify the
    transferred value. The modifiers should either be children of
    :ref:`Modifier` or callables taking a :obj:`dict` as argument and
    returning a :obj:`dict`.
  """

  _count = 0

  def __init__(self,
               input_block,
               output_block,
               modifiers: Optional[List[Union[ModifierType, Modifier]]] = None,
               name: Optional[str] = None) -> None:
    """Sets the instance attributes.

    Args:
      input_block: The Block sending data through the link.
      output_block: The Block receiving data through the link.
      modifiers: A :obj:`list` containing children of :ref:`Modifier` and/or
        callables. If several objects given ,they will be called in the given
        order. See :ref:`Modifiers` for more information.
      name: Name of the link, to differentiate it from the others when
        debugging. If no specific name is given, the links are anyway numbered
        in the order in which they are instantiated in the code.
    """

    if modifiers is None:
      modifiers = list()

    self.name = name if name is not None else f'link{self._get_count()}'
    self._in, self._out = Pipe()
    self._modifiers = modifiers

    # Associating the link to the input and output blocks
    input_block.add_output(self)
    output_block.add_input(self)

    self._last_warn = time()
    self._logger: Optional[logging.Logger] = None

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
    """

    if self._logger is None:
      self._logger = logging.getLogger(
        f"{current_process().name}.{self.name}")

    self._logger.log(log_level, msg)

  def poll(self) -> bool:
    """Returns :obj:`True` if there's data available for reading."""

    return self._in.poll()

  def send(self, value: Dict[str, Any]) -> None:
    """Sends a value from the upstream Block to the downstream Block.

    Before sending, applies the given modifiers and makes sure there's room in
    the Pipe for sending the data.
    """

    for mod in self._modifiers:
      # Case when the modifier is a class
      if hasattr(mod, 'evaluate'):
        value = mod.evaluate(deepcopy(value))
      # Case when the modifier is a method
      else:
        value = mod(deepcopy(value))

    if value is None:
      return

    if not isinstance(value, dict):
      self.log(logging.ERROR, f"Trying to send object of type {type(value)} "
                              f"instead of dict !")
      raise LinkDataError

    # Finally, sending the dict to the link
    if select([], [self._out], [], 0)[1]:
      self._out.send(value)
    else:
      if time() - self._last_warn > 1:
        self._last_warn = time()
        self.log(logging.WARNING, f"Cannot send the values, the Link is "
                                  f"full !")

  def recv(self) -> Dict[str, Any]:
    """Reads a single value from the Link and returns it.

    The read value is the oldest available in the Link, see :meth:`recv_last`
    for reading the newest available value.

    If no data is available in the Link, returns an empty :obj:`dict`.

    Returns:
      A :obj:`dict` whose keys are the labels being sent, and for each key a
      single value (usually a :obj:`float` or a :obj:`str`).
    """

    if self._in.poll():
      return self._in.recv()
    else:
      return dict()

  def recv_last(self) -> Dict[str, Any]:
    """Reads all the available values in the Link, and returns the newest one.

    If no data is available in the Link, returns an empty :obj:`dict`. All the
    data that is not returned is permanently dropped.

    Returns:
      A :obj:`dict` whose keys are the labels being sent, and for each key a
      single value (usually a :obj:`float` or a :obj:`str`).
    """

    data = dict()

    while self._in.poll():
      data = self._in.recv()

    return data

  def recv_chunk(self) -> Dict[str, List[Any]]:
    """Reads all the available values in the Link, and returns them all.

    Returns:
      A :obj:`dict` whose keys are the labels being sent, and for each key a
      :obj:`list` of the received values. The first item in the list is the
      oldest one available in the Link, the last item is the newest available.
    """

    ret = defaultdict(list)

    while self._in.poll():
      data = self._in.recv()
      for label, value in data.items():
        ret[label].append(value)

    return dict(ret)


def link(in_block,
         out_block,
         modifier: Optional[Union[List[Union[ModifierType, Modifier]],
                                  Union[ModifierType, Modifier]]] = None,
         name: Optional[str] = None) -> None:
  """Function linking two blocks, allowing to send data from one to the other.

  The created link is unidirectional, from the input block to the output block.
  Under the hood, a link is basically a :class:`multiprocessing.Pipe` with
  extra features.

  Args:
    in_block: The Block sending data through the link.
    out_block: The Block receiving data through the link.
    modifier: Either a child class of :ref:`Modifier`, or a callable, or a
      :obj:`list` containing such objects. If several given (in a list), calls
      them in the  given order. See :ref:`Modifiers` for more information.
    name: Name of the link, to differentiate it from the others when debugging.
      If no specific name is given, the links are anyway numbered in the order
      in which they are instantiated in the code.
  """

  # Forcing the modifiers into lists
  if modifier is not None and not isinstance(modifier, list):
    modifier = [modifier]

  # Actually creating the Link object
  Link(input_block=in_block,
       output_block=out_block,
       modifiers=modifier,
       name=name)
