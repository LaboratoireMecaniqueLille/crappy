# coding: utf-8

# Todo:
#   Allow to actually set the timeout for sending
#   Allow to set the timeout for receiving
#   Assign different names to the links with a class attribute


from multiprocessing import Pipe
from time import time
from threading import Thread
from copy import copy
from functools import wraps
from typing import Callable, Union, Any, Dict, NoReturn, Optional, Literal, \
  List

from .._global import CrappyStop
from ..modifier import Modifier


def error_if_string(recv: Callable) -> Callable:
  """Decorator to raise an error if the function returns a string."""

  @wraps(recv)
  def wrapper(*args, **kwargs):
    ret = recv(*args, **kwargs)
    if isinstance(ret, str):
      raise CrappyStop
    return ret
  return wrapper


class MethodThread(Thread):
  """ThreadMethodThread, daemonic descendant class of :mod:`threading`.

  Thread which simply runs the specified target method with the specified
  arguments.
  """

  def __init__(self, target: Callable, args, kwargs) -> None:
    Thread.__init__(self)
    self.setDaemon(True)
    self.target, self.args, self.kwargs = target, args, kwargs
    self.start()

  def run(self) -> None:
    try:
      self.result = self.target(*self.args, **self.kwargs)
    except Exception as e:
      self.exception = e
    else:
      self.exception = None


def win_timeout(timeout: float = None) -> Callable:
  """Decorator for adding a timeout to a link send."""

  def win_timeout_proxy(f: Callable) -> Callable:
    @wraps(f)
    def wrapper(*args, **kwargs) -> Any:
      worker = MethodThread(f, args, kwargs)
      if timeout is None:
        return worker
      worker.join(timeout)
      if worker.is_alive():
        raise TimeoutError("timeout error in pipe send")
      elif worker.exception is not None:
        raise worker.exception
      else:
        return worker.result

    return wrapper

  return win_timeout_proxy


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

  def __init__(self,
               input_block=None,
               output_block=None,
               conditions: List[Union[Callable, Modifier]] = None,
               modifiers: List[Union[Callable, Modifier]] = None,
               timeout: float = 0.1,
               action: Literal['warn', 'kill', 'NoWarn'] = "warn",
               name: str = "link") -> None:
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
      name (:obj:`str`, optional): Name of the link, to differentiate it from
        the others when debugging.
    """

    # For compatibility (condition is deprecated, use modifier)
    if conditions is not None:
      if modifiers is not None:
        modifiers += conditions
      else:
        modifiers = conditions

    # Setting the attributes
    self.name = name
    self.in_, self.out_ = Pipe()
    self.external_trigger = None
    self.modifiers = modifiers
    self.timeout = timeout
    self.action = action

    if input_block is not None and output_block is not None:
      input_block.add_output(self)
      output_block.add_input(self)

  def close(self) -> None:
    self.in_.close()
    self.out_.close()

  def send(self, value: Union[dict, str]) -> None:
    """Sends the value, or a modified value if you pass it through a modifier.
    """

    try:
      self.send_timeout(value)
    except TimeoutError as e:
      if self.action == "warn":
        print(
            "WARNING : Timeout error in pipe send! Link name: %s" % self.name)
      elif self.action == "kill":
        print("Killing Link : ", e)
        raise
      elif self.action == "NoWarn":
        pass
      else:  # for debugging !!
        print(self.action)
    except Exception as e:
      print("Exception in link send %s : %s " % (self.name, str(e)))
      raise

  @win_timeout(1)
  def send_timeout(self, value: dict) -> None:
    """Method for sending data with a given timeout on the link."""

    try:
      if self.modifiers is None or isinstance(value, str):
        self.out_.send(value)
      else:
        for mod in self.modifiers:
          if hasattr(mod, 'evaluate'):
            value = mod.evaluate(copy(value))
          else:
            value = mod(copy(value))
          if value is None:
            break
        if value is not None:
          self.out_.send(value)
    except Exception as e:
      print("Exception in link %s : %s " % (self.name, str(e)))
      if not self.out_.closed:
        self.out_.send('close')
        self.out_.close()
      raise

  @error_if_string  # Recv will raise an error if a string is received
  def recv(self, blocking: bool = True) -> Dict[str, list]:
    """Receives data.

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
      if blocking or self.in_.poll():
        return self.in_.recv()

    except TimeoutError as e:
      if self.action == "warn":
        print(
            "WARNING : Timeout error in pipe send! Link name: %s" % self.name)
      elif self.action == "kill":
        print("Killing Link %s : %s" % (self.name, str(e)))
        raise
      elif self.action == "NoWarn":
        pass
      else:  # for debugging !!
        print(self.action)
    except Exception as e:
      print("EXCEPTION in link %s : %s " % (self.name, str(e)))
      if not self.in_.closed:
        self.in_.close()
      raise

  def poll(self) -> bool:
    return self.in_.poll()

  def clear(self) -> NoReturn:
    while self.in_.poll():
      self.in_.recv_bytes()

  def recv_last(self, blocking: bool = False) -> Dict[str, list]:
    """Returns only the LAST value in the pipe, dropping all the others.

    Note:
      If ``blocking`` is :obj:`False`, will return :obj:`None` if there is no
      data waiting.

      If ``blocking`` is :obj:`True`, will wait for at least one data.

    Warning:
      Unlike :meth:`recv`, default is NON blocking.
    """

    if blocking:
      data = self.recv()
    else:
      data = None
    while self.in_.poll():
      data = self.recv()
    return data

  def recv_chunk(self, length: int = 0) -> Dict[str, list]:
    """Allows you to receive a chunk of data.

    If ``length`` `> 0` it will return a :obj:`dict` containing :obj:`list` of
    the last length received data.

    If ``length`` `= 0`, it will return all the waiting data until the pipe is
    empty.

    If the pipe is already empty, it will wait to return at least one value.
    """

    ret = self.recv()
    for k in ret:
      ret[k] = [ret[k]]
    c = 0
    while c < length or (length <= 0 and self.poll()):
      c += 1
      try:
        data = self.recv()  # Here, we need to send our data first
      except CrappyStop:
        self.out_.send("stop")  # To re-raise on next call
        break
      for k in ret:
        try:
          ret[k].append(data[k])
        except KeyError:
          raise IOError(str(self) + " Got data without label " + k)
    return ret

  def recv_delay(self, delay: float) -> dict:
    """Useful for blocks that don't need data all so frequently.

    It will continuously receive data for a given delay and return them as a
    single :obj:`dict` containing :obj:`list` of the values.

    Note:
      All the :meth:`recv` calls are blocking so this method will take AT LEAST
      ``delay`` seconds to return, but it could be more since it may wait for
      data.

      Also, it will return at least one reading.
    """

    t = time()
    ret = self.recv()  # If we get CrappyStop at this instant, no data loss
    for k in ret:
      ret[k] = [ret[k]]
    while time() - t < delay:
      try:
        data = self.recv()  # Here, we need to send our data first
      except CrappyStop:
        self.out_.send("stop")  # To re-raise on next call
        break
      for k in ret:
        try:
          ret[k].append(data[k])
        except KeyError:
          raise IOError(str(self) + " Got data without label " + k)
    return ret

  def recv_chunk_nostop(self) -> Union[dict, None]:
    """Experimental feature, to be used in :meth:`finish` methods to recover
    the final remaining data (possibly after a stop signal)."""

    lst = []
    while self.in_.poll():
      data = self.in_.recv()
      if isinstance(data, dict):
        lst.append(data)
    r = {}
    for d in lst:
      for k, v in d.items():
        if k in r:
          r[k].append(v)
        else:
          r[k] = [v]
    return r


def link(in_block,
         out_block,
         condition: Optional[Union[List[Union[Modifier, Callable]],
                                   Union[Modifier, Callable]]] = None,
         modifier: Optional[Union[List[Union[Modifier, Callable]],
                                  Union[Modifier, Callable]]] = None,
         timeout: float = 0.1,
         action: Literal['warn', 'kill', 'NoWarn'] = "warn",
         name: str = "link") -> None:
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
    name (:obj:`str`, optional): Name of the link, to differentiate it from
      the others when debugging.
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
