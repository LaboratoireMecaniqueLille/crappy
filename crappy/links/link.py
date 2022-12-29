# coding: utf-8

from multiprocessing import Pipe
from time import time, sleep
from copy import deepcopy
from typing import Callable, Union, Any, Dict, Optional, List
from collections import defaultdict
from select import select

from ..modifier import Modifier

Modifier_type = Callable[[Dict[str, Any]], Dict[str, Any]]


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
               modifiers: Optional[List[Union[Modifier_type,
                                              Modifier]]] = None,
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

  def __new__(cls, *args, **kwargs):
    """When instantiating a new Link, increments the Link counter."""

    cls._count += 1
    return super().__new__(cls)

  @classmethod
  def _get_count(cls) -> int:
    """Returns the current number of instantiates Links, as an :obj:`int`."""

    return cls._count

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
      print(f"Warning in Link {self.name}: trying to send object of type "
            f"{type(value)} instead of dict, not sending !")

    # Finally, sending the dict to the link
    if select([], [self._out], [], 0)[1]:
      self._out.send(value)
    else:
      if time() - self._last_warn > 1:
        self._last_warn = time()
        print(f"Cannot send values in Link {self.name}, the Link is full !\n"
              f"Maybe the values are not being read by the downstream block, "
              f"or too much data is being sent.")

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

  def recv_chunk(self,
                 delay: Optional[float] = None,
                 poll_delay: float = 0.01) -> Dict[str, list]:
    """Reads all the available values in the Link, and returns them all.

    Args:
      delay: If given specifies a delay, as a :obj:`float`, during which the
        method acquired data before returning. All the data received during
        this delay is saved and returned. Otherwise, just reads all the
        available data and returns as soon as it is exhausted.
      poll_delay: If the ``delay`` argument is given, corresponds to the time
        to sleep between two Link polls. It ensures that the method doesn't
        spam the CPU in vain.

    Returns:
      A :obj:`dict` whose keys are the labels being sent, and for each key a
      :obj:`list` of the received values. The first item in the list is the
      oldest one available in the Link, the last item is the newest available.
    """

    ret = defaultdict(list)
    t_init = time()

    # Case when no delay is specified
    if delay is None:
      while self._in.poll():
        data = self._in.recv()
        for label, value in data.items():
          ret[label].append(value)

    # Case when a delay is specified
    else:
      # Looping until the delay is exhausted
      while time() - t_init < delay:
        if self._in.poll():
          data = self._in.recv()
          for label, value in data.items():
            ret[label].append(value)
        else:
          sleep(poll_delay)

    return dict(ret)


def link(in_block,
         out_block,
         modifier: Optional[Union[List[Union[Modifier_type, Modifier]],
                                  Union[Modifier_type, Modifier]]] = None,
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
