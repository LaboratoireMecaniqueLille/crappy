##  @defgroup links Links
# Link class. All connection between Blocks should be made with this.
# @{

##  @defgroup init Init
# @{

## @file __init__.py
# @brief  Link class. All connection between Blocks should be made with this.
#
# @authors Corentin Martel, Robin Siemiatkowski, Victor Couty
# @version 0.2
# @date 19/01/2017
from __future__ import print_function

from multiprocessing import Pipe
import copy
from time import time
from threading import Thread

def error_if_stop(recv):
  "Decorator to raise an error if the function returns a string"
  def wrapper(*args,**kwargs):
    ret = recv(*args,**kwargs)
    if type(ret) == str:
      raise KeyboardInterrupt
    return ret
  return wrapper

class TimeoutError(Exception):
  """Custom error to raise in case of timeout."""
  pass


class MethodThread(Thread):
  """
  ThreadMethodThread, daemonic descendant class of threading.Thread which
  simply runs the specified target method with the specified arguments.
  """

  def __init__(self, target, args, kwargs):
    Thread.__init__(self)
    self.setDaemon(True)
    self.target, self.args, self.kwargs = target, args, kwargs
    self.start()

  def run(self):
    try:
      self.result = self.target(*self.args, **self.kwargs)
    except Exception, e:
      self.exception = e
    except:
      self.exception = Exception()
    else:
      self.exception = None


def win_timeout(timeout=None):
  """Decorator for adding a timeout to a link send."""

  def win_timeout_proxy(f):
    def wrapper(*args, **kwargs):
      worker = MethodThread(f, args, kwargs)
      if timeout is None:
        return worker
      worker.join(timeout)
      if worker.isAlive():
        raise TimeoutError("timeout error in pipe send")
      elif worker.exception is not None:
        raise worker.exception
      else:
        return worker.result

    return wrapper

  return win_timeout_proxy


class Link(object):
  """
  Link class. All connection between Blocks should be made with this.
  """

  def __init__(self, input_block=None, output_block=None, condition=[],
                                    timeout=0.1, action="warn", name="link"):
    """
    Creates a pipe and is used to transfer information between Blocks.
    You can add one or multiple conditions to modify the value transfered.


    Args:
        name: name of a link to recognize it on timeout.
        condition : Children class of links.Condition, optionnal
            Each "send" call will pass through the condition.evaluate method
            and sends the returned value.
            If condition has not evaluate method, send will try call condition.
            You can pass a list of conditions, they will be executed in order.

        timeout : int or float, default = 0.1
            Timeout for the send method.

        action : {'warn','kill','NoWarn',str}, default = "warn"
            Action in case of TimeoutError in the send method. You can warn
            only, not warn or choose to kill the link. If any other string,
            will be printed in case of error to debug.
    """
    self.name = name
    self.in_, self.out_ = Pipe(duplex=False)
    self.external_trigger = None
    if type(condition) is list:
      self.condition = condition
    else:
      self.condition = [condition]
    self.timeout = timeout
    self.action = action
    if not None in [input_block, output_block]:
      input_block.add_output(self)
      output_block.add_input(self)

  def add_external_trigger(self, link_instance):
    """Add an external trigger Link."""
    self.external_trigger = link_instance
    for cond in self.condition:
      cond.external_trigger = link_instance

  def close(self):
    self.in_.close()
    self.out_.close()

  def send(self, value):
    """
    Send the value, or a modified value if you pass it through a
    condition.
    """
    try:
      self.send_timeout(value)
    except TimeoutError as e:
      if self.action == "warn":
        print("WARNING : Timeout error in pipe send! Link name: %s" % self.name)
      elif self.action == "kill":
        print("Killing Link : ", e)
        raise
      elif self.action == "NoWarn":
        pass
      else:  # for debugging !!
        print(self.action)
    except Exception as e:
      print("Exception in link send %s : %s " % (self.name, e.message))
      raise

  @win_timeout(1)
  def send_timeout(self, value):
    try:
      if self.condition is None:
        self.out_.send(value)
      else:
        for cond in self.condition:
          if hasattr(cond,'evaluate'):
            value = cond.evaluate(copy.copy(value))
          else:
            value = cond(copy.copy(value))
        if value is not None:
          self.out_.send(value)
    except KeyboardInterrupt:
      print("Keyboard interrupt received in link: " % self.name)
      raise
    except Exception as e:
      print("Exception in link %s : %s " % (self.name, e.message))
      if not self.out_.closed:
        self.out_.send('close')
        self.out_.close()
      raise Exception(e)

  @error_if_stop # Recv will raise an error if 'stop' is recved
  def recv(self, blocking=True):
    """
    Receive data. If blocking=False, return None if there is no data

    Args:
        blocking: Enable (True) or disable (False) blocking mode.

    Returns:
        If blocking is True, recv() method will wait
        until data is available on the pipe and return the received data,
        otherwise, it will check if there is data available to return,
        or return None if the pipe is empty.
    """
    try:
      if blocking:
        return self.in_.recv()
      else:
        if self.in_.poll():
          return self.in_.recv()

    except TimeoutError as e:
      if self.action == "warn":
        print("WARNING : Timeout error in pipe send! Link name: %s" % self.name)
      elif self.action == "kill":
        print("Killing Link %s : %s" % (self.name, e.message))
        raise
      elif self.action == "NoWarn":
        pass
      else:  # for debugging !!
        print(self.action)
    except Exception as e:
      print("EXCEPTION in link %s : %s " % (self.name, e.message))
      if not self.in_.closed:
        self.in_.close()
      raise

  def poll(self):
    return self.in_.poll()

  def clear(self):
    while self.in_.poll():
      self.in_.recv()

  def recv_last(self,blocking=False):
    """Returns only the LAST value in the pipe, dropping all the others.
    if blocking=False: will return None if there is no data waiting
    if blocking=True: Will wait for at least one data
    Warning! Unlike recv, default is NON blocking"""
    if blocking:
      data = self.in_.recv()
    else:
      data = None
    while self.in_.poll():
      data = self.in_.recv()
    return data

  def recv_chunk(self,length=0):
    """
    Allows you to receive a chunk of data: if length > 0 it will return an
    OrderedDict containing LISTS of the last length received data
    If length=0, it will return all the waiting data until the pipe is empty
    if the pipe is already empty, it will wait to return at least one value.
    """
    ret = self.recv()
    for k in ret:
      ret[k] = [ret[k]]
    c = 0
    while c < length or (length <= 0 and self.poll()):
      c += 1
      data = self.recv()
      for k in ret:
        try:
          ret[k].append(data[k])
        except KeyError:
          raise IOError(str(self)+" Got data without label "+k)
    return ret

  def recv_delay(self,delay=1):
    """
    Useful for blocks that don't need data all so frequently:
    It will continuously receive data for a given delay and return them as a
    single OrderedDict containing lists of the values.
    Note that all the .recv calls are blocking so this method will take
    AT LEAST delay seconds to return, but it could be more since it may wait
    for data. Also, it will return at least one reading.
    """
    t = time()
    ret = self.recv()
    for k in ret:
      ret[k] = [ret[k]]
    while time()-t < delay:
      data = self.recv()
      for k in ret:
        try:
          ret[k].append(data[k])
        except KeyError:
          raise IOError(str(self)+" Got data without label "+k)
    return ret

def link(in_block, out_block, **kwargs):
  """
  Function that links two blocks
  For the object, see Link
  """
  Link(input_block=in_block, output_block=out_block, **kwargs)
