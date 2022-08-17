# coding: utf-8

from multiprocessing import Pipe
from time import time
from threading import Thread
from copy import copy
from typing import Callable, Union, Any, Dict, Optional, List

from .._global import CrappyStop
from ..modifier import Modifier


class Link:
  """This class is used for transferring information between the blocks.

  The created link is unidirectional, from the input block to the output block.
  Under the hood, a link is basically a :class:`multiprocessing.Pipe` with
  extra features.

  Note:
    You can add one or multiple :ref:`Modifiers` to modify the transferred
    value. The modifiers should either be children of :ref:`Modifier` or
    callables taking a :obj:`dict` as argument and returning a :obj:`dict`.
  """

  count = 0

  def __init__(self,
               input_block=None,
               output_block=None,
               conditions: List[Union[Callable, Modifier]] = None,
               modifiers: List[Union[Callable, Modifier]] = None,
               timeout: float = 1,
               action: str = "warn",
               name: Optional[str] = None) -> None:
    """Sets the instance attributes.

    Args:
      input_block: The Block sending data through the link.
      output_block: The Block receiving data through the link.
      conditions: Deprecated, kept only for backward-compatibility.
      modifiers: A :obj:`list` containing children of :ref:`Modifier` and/or
        callables. If several objects given ,they will be called in the given
        order. See :ref:`Modifiers` for more information.
      timeout: Sets a timeout for sending the data in the link.
      action: Action to perform in case of a :exc:`TimeoutError` during the
        :meth:`send` method. Should be in:
        ::

          'warn', 'kill', 'NoWarn',

        any other value would be for debugging.
      name: Name of the link, to differentiate it from the others when
        debugging. If no specific name is given, the links are anyway numbered
        in the order in which they are instantiated in the code.
    """

    # For compatibility (condition is deprecated, use modifier)
    if conditions is not None:
      if modifiers is not None:
        modifiers += conditions
      else:
        modifiers = conditions

    # Setting the attributes
    count = self._count_links()
    self.name = name if name is not None else f'link{count}'
    self._in, self._out = Pipe()
    self._modifiers = modifiers
    self._action = action
    self._timeout = timeout

    # Associating the link with the input and output blocks if they are given
    if input_block is not None and output_block is not None:
      input_block.add_output(self)
      output_block.add_input(self)

  @classmethod
  def _count_links(cls) -> int:
    """Simply increments the link count and returns it."""

    cls.count += 1
    return cls.count

  def send(self, value: Union[Dict[str, Any], str]) -> None:
    """Sends a value through the link.

    In case of a timeout exception, executes the user-defined action. Raises
    any other exception caught.
    """

    # Trying to send a value through a link
    try:
      send_job = Thread(target=self._send_timeout, args=(value,), daemon=True)
      send_job.start()

      # Waits for the thread to return
      send_job.join(self._timeout)

      # If it's taking too long, raising an exception
      if send_job.is_alive():
        raise TimeoutError

    # If a timeout exception is raised, handling it according to the action
    except TimeoutError as exc:
      # Warning the user
      if self._action == "warn":
        print(f"WARNING : Timeout error in pipe send! Link name: {self.name}")
      # Stopping the program
      elif self._action == "kill":
        raise TimeoutError("Killing Link : " + str(exc))
      # Simply ignoring
      elif self._action == "NoWarn":
        pass
      # Printing the user-defined message
      else:
        print(self._action)

    # Raising any other caught exception
    except Exception as exc:
      print(f"Exception in link send {self.name} : {str(exc)}")
      raise

  def _send_timeout(self, value: Union[Dict[str, Any], str]) -> None:
    """Method for sending data with a given timeout on the link."""

    try:
      # Sending if the value is None or a string
      if self._modifiers is None or isinstance(value, str):
        self._out.send(value)

      # Else, first applying the modifiers
      else:
        for mod in self._modifiers:
          # Case when the modifier is a class
          if hasattr(mod, 'evaluate'):
            value = mod.evaluate(copy(value))
          # Case when the modifier is a method
          else:
            value = mod(copy(value))
          # Exiting the for loop if nothing left in the dict to send
          if value is None:
            break

        # Finally, sending the dict to the link
        if value is not None:
          self._out.send(value)

    # Raising any exception caught, but first sending a stop message downstream
    except Exception as exc:
      print(f"Exception in link {self.name} : {str(exc)}")
      if not self._out.closed:
        self._out.send('close')
        self._out.close()
      raise

  def recv(self, blocking: bool = True) -> Optional[Dict[str, Any]]:
    """Receives data from a link and returns it as a dict.

    Note:
      If ``blocking`` is :obj:`False`, returns :obj:`None` if there is no
      data.

    Args:
      blocking (:obj:`bool`, optional): Enables (:obj:`True`) or disables
        (:obj:`False`) blocking mode.

    Returns:
      If ``blocking`` is :obj:`True`, :meth:`recv` method will wait until data
      is available in the pipe and return the received data. Otherwise, it will
      check if there is data available to return, or return :obj:`None` if the
      pipe is empty.
    """

    try:
      if blocking or self.poll():
        # Simply collecting the data to receive
        ret = self._in.recv()

        # Raising a CrappyStop in case a string is received
        if isinstance(ret, str):
          raise CrappyStop

        return ret

    # If a timeout exception is raised, handling it according to the action
    except TimeoutError as exc:
      # Warning the user
      if self._action == "warn":
        print(f"WARNING : Timeout error in pipe recv! Link name: {self.name}")
      # Stopping the program
      elif self._action == "kill":
        raise TimeoutError(f"Killing Link : {exc}")
      # Simply ignoring
      elif self._action == "NoWarn":
        pass
      # Printing the user-defined message
      else:
        print(self._action)

    # Raising any CrappyStop
    except CrappyStop:
      raise

    # Raising any other caught exception and displaying message
    except Exception as exc:
      print(f"Exception in link recv {self.name} : {str(exc)}")
      raise

  def poll(self) -> bool:
    """Simple wrapper telling whether there's data in the link or not."""

    return self._in.poll()

  def clear(self) -> None:
    """Flushes the link."""

    while self.poll():
      self._in.recv_bytes()

  def recv_last(self, blocking: bool = False) -> Optional[Dict[str, Any]]:
    """Returns only the last value in the pipe, dropping all the others.

    Note:
      If ``blocking`` is :obj:`False`, will return :obj:`None` if there is no
      data waiting.

      If ``blocking`` is :obj:`True`, will wait for at least one data.

    Warning:
      Unlike :meth:`recv`, default is non blocking.
    """

    # First, block if necessary
    data = self.recv(blocking)
    if data is None:
      return

    # Then, flush the pipe and keep only the last value
    while True:
      new = self.recv(blocking=False)
      if new is None:
        return data
      data = new

  def recv_chunk(self,
                 blocking: bool = True) -> Optional[Dict[str, List[Any]]]:
    """Returns all the data waiting in a link.

    Note:
      If ``blocking`` is :obj:`False`, will return :obj:`None` if there is no
      data waiting.

      If ``blocking`` is :obj:`True`, will wait for at least one data.
    """

    # First, block if necessary and return if the link is empty
    ret = self.recv(blocking)
    if ret is None:
      return

    # Putting the received values in lists
    for label, value in ret.items():
      ret[label] = [value]

    while True:
      try:
        data = self.recv(blocking=False)

      # Sending a stop message if a CrappyStop is raised
      except CrappyStop:
        self._out.send("stop")
        return ret

      # Return when the link is empty
      if data is None:
        return ret

      # Adding the received data to the created lists
      for label in ret:
        try:
          ret[label].append(data[label])
        # Raising an exception in case a label is missing
        except KeyError:
          raise IOError(f"{str(self)} Got data without label {label}")

  def recv_delay(self, delay: float) -> Dict[str, List[Any]]:
    """Same as :meth:`recv_chunk` except it runs for a given delay no matter
    if the link is empty or not.

    Useful for blocks with a low looping frequency that don't need data so
    frequently.

    Note:
      All the :meth:`recv` calls are blocking so this method will take at least
      ``delay`` seconds to return, but it could be more since it may wait for
      data.

      Also, it will return at least one reading.
    """

    t_init = time()
    # This first call to recv is blocking
    ret = self.recv(blocking=True)

    # Putting the received values in lists
    for label, value in ret.items():
      ret[label] = [value]

    while time() - t_init < delay:
      try:
        data = self.recv(blocking=True)

      # Sending a stop message if a CrappyStop is raised
      except CrappyStop:
        self._out.send("stop")
        break

      # Adding the received data to the created lists
      for label in ret:
        try:
          ret[label].append(data[label])
        # Raising an exception in case a label is missing
        except KeyError:
          raise IOError(f"{str(self)} Got data without label {label}")

    return ret

  def recv_chunk_no_stop(self) -> Optional[Dict[str, List[Any]]]:
    """Experimental feature, to be used in :meth:`finish` methods to recover
    the final remaining data (possibly after a stop signal)."""

    # First, collecting all the remaining data
    recv = []
    while self.poll():
      data = self._in.recv()
      if isinstance(data, dict):
        recv.append(data)

    # Then, organizing it into a nice dict to return
    ret = {}
    for data in recv:
      for label, value in data.items():
        if label in ret:
          ret[label].append(value)
        else:
          ret[label] = [value]

    return ret if ret else None


def link(in_block,
         out_block,
         condition: Optional[Union[List[Union[Modifier, Callable]],
                                   Union[Modifier, Callable]]] = None,
         modifier: Optional[Union[List[Union[Modifier, Callable]],
                                  Union[Modifier, Callable]]] = None,
         timeout: float = 1,
         action: str = "warn",
         name: Optional[str] = None) -> None:
  """Function linking two blocks, allowing to send data from one to the other.

  The created link is unidirectional, from the input block to the output block.
  Under the hood, a link is basically a :class:`multiprocessing.Pipe` with
  extra features.

  Args:
    in_block: The Block sending data through the link.
    out_block: The Block receiving data through the link.
    condition: Deprecated, kept only for backward-compatibility.
    modifier: Either a child class of :ref:`Modifier`, or a callable, or a
      :obj:`list` containing such objects. If several given (in a list), calls
      them in the  given order. See :ref:`Modifiers` for more information.
    timeout: Sets a timeout for sending the data in the link.
    action: Action to perform in case of a :exc:`TimeoutError` during the
      :meth:`send` method. Should be in:
      ::

        'warn', 'kill', 'NoWarn',

      any other value would be for debugging.
    name: Name of the link, to differentiate it from the others when debugging.
      If no specific name is given, the links are anyway numbered in the order
      in which they are instantiated in the code.
  """

  # Forcing the conditions and modifiers into lists
  if condition is not None and not isinstance(condition, list):
    condition = [condition]
  if modifier is not None and not isinstance(modifier, list):
    modifier = [modifier]

  # Actually creating the Link object
  Link(input_block=in_block,
       output_block=out_block,
       conditions=condition,
       modifiers=modifier,
       timeout=timeout,
       action=action,
       name=name)
