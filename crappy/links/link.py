# coding: utf-8

# Link class. All connection between Blocks should be made with this.


from multiprocessing import Pipe
from time import time
from threading import Thread
from copy import copy
from functools import wraps
from typing import Callable, Union

from .._global import CrappyStop


def error_if_string(recv: Callable) -> Callable:
  """Decorator to raise an error if the function returns a string."""

  @wraps(recv)
  def wrapper(*args, **kwargs):
    ret = recv(*args, **kwargs)
    if isinstance(ret, str):
      raise CrappyStop
    return ret
  return wrapper


class TimeoutError(Exception):
  """Custom error to raise in case of timeout."""

  pass


class MethodThread(Thread):
  """ThreadMethodThread, daemonic descendant class of :mod:`threading`.

  Thread which simply runs the specified target method with the specified
  arguments.
  """

  def __init__(self, target, args, kwargs):
    Thread.__init__(self)
    self.setDaemon(True)
    self.target, self.args, self.kwargs = target, args, kwargs
    self.start()

  def run(self):
    try:
      self.result = self.target(*self.args, **self.kwargs)
    except Exception as e:
      self.exception = e
    else:
      self.exception = None


def win_timeout(timeout: float = None):
  """Decorator for adding a timeout to a link send."""

  def win_timeout_proxy(f: Callable) -> Callable:
    @wraps(f)
    def wrapper(*args, **kwargs):
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


class Link(object):
  """Link class. All connections between Blocks should be made with this.

  It creates a pipe and is used to transfer information between Blocks.

  Note:
    You can add one or multiple :ref:`Modifiers` to modify the transferred
    value.
  """

  def __init__(self,
               input_block=None,
               output_block=None,
               condition=None,
               modifier=None,
               timeout: float = 0.1,
               action: str = "warn",
               name: str = "link") -> None:
    """Sets the instance attributes.

    Args:
      input_block:
      output_block:
      condition: Children class of :class:`links.Condition`, for backward
        compatibility only. Can be a single condition or a :obj:`list` of
        conditions that will be executed in the given order.
      modifier:
      timeout (:obj:`float`, optional): Timeout for the :meth:`send` method.
      action (:obj:`str`, optional): Action to perform in case of a
        :exc:`TimeoutError` during the :meth:`send` method. Should be in:
        ::

          'warn', 'kill', 'NoWarn',

        any other value would be for debugging.
      name (:obj:`str`, optional): Name of a link to recognize it on timeout.
    """

    # For compatibility (condition is deprecated, use modifier)
    if condition is not None:
      modifier = condition
    # --
    self.name = name
    self.in_, self.out_ = Pipe()
    self.external_trigger = None
    if modifier is not None:
      self.modifiers = modifier if isinstance(modifier, list) else [modifier]
    else:
      self.modifiers = None
    self.timeout = timeout
    self.action = action
    if None not in [input_block, output_block]:
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
  def recv(self, blocking: bool = True) -> Union[dict, None]:
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

  def clear(self) -> None:
    while self.in_.poll():
      self.in_.recv_bytes()

  def recv_last(self, blocking: bool = False) -> Union[dict, None]:
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

  def recv_chunk(self, length: int = 0) -> dict:
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


def link(in_block, out_block, **kwargs) -> None:
  """Function that links two blocks.

  Note:
    For the object, see :ref:`Link`.
  """

  Link(input_block=in_block, output_block=out_block, **kwargs)
