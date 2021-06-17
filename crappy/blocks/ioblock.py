# coding: utf-8

from .block import Block
from ..inout import inandout_dict, in_dict, out_dict, inout_dict


class IOBlock(Block):
  """This block is used to communicate with :ref:`In / Out` objects.

  They can be used as sensor, actuators or both.
  """

  def __init__(self,
               name,
               freq=None,
               verbose=False,
               labels=None,
               cmd_labels=None,
               trigger=None,
               streamer=False,
               initial_cmd=0,
               exit_values=None,
               **kwargs):
    """Sets the args and initializes the parent class.

    Args:
      name (:obj:`str`): The name of the :ref:`In / Out` class to instantiate.
      freq (:obj:`float`, optional): The looping frequency of the block, if
        :obj:`None` will go as fast as possible.
      verbose (:obj:`bool`, optional): Prints extra information if :obj:`True`.
      labels (:obj:`list`, optional): A :obj:`list` of the output labels.
      cmd_labels (:obj:`list`, optional): The :obj:`list` of the labels
        considered as inputs for this block. Will call :meth:`set_cmd`  in the
        :ref:`In / Out` object with the values received on this labels.
      trigger (:obj:`int`, optional): If the block is triggered by another
        block, this must specify the index of the input considered as a
        trigger. The data going through this link is discarded, add another
        link if the block should also consider it as an input.
      streamer (:obj:`bool`, optional): If :obj:`False`, will call
        :meth:`get_data` else, will call :meth:`get_stream` in the
        :ref:`In / Out` object (only if it has these methods, of course).
      initial_cmd (:obj:`list`, optional): The initial values for the outputs,
        sent during :meth:`prepare`. If it is a single value, then it will send
        this same value for all the output labels.
      exit_values (:obj:`list`, optional): If not :obj:`None`, the outputs will
        be set to these values when Crappy is ending (or crashing).
      **kwargs: The arguments to be passed to the :ref:`In / Out` class.
    """

    Block.__init__(self)
    self.niceness = -10
    self.freq = freq
    self.verbose = verbose
    self.labels = labels
    self.cmd_labels = [] if cmd_labels is None else cmd_labels
    self.trigger = trigger
    self.streamer = streamer
    self.initial_cmd = initial_cmd
    self.exit_values = exit_values

    if self.labels is None:
      if self.streamer:
        self.labels = ['t(s)', 'stream']
      else:
        self.labels = ['t(s)'] + \
                      [str(c) for c in kwargs.get("channels", ['1'])]
    self.device_name = name.capitalize()
    self.device_kwargs = kwargs
    self.stream_idle = True
    if not isinstance(self.initial_cmd, list):
      self.initial_cmd = [self.initial_cmd] * len(self.cmd_labels)
    if not isinstance(self.exit_values, list) and self.exit_values is not None:
      self.exit_values = [self.exit_values] * len(self.cmd_labels)
    if self.exit_values is not None:
      assert len(self.exit_values) == len(self.cmd_labels),\
          'Invalid number of exit values!'
    self.device = inout_dict[self.device_name](**self.device_kwargs)

  def prepare(self):
    self.to_get = list(range(len(self.inputs)))
    if self.trigger is not None:
      self.to_get.remove(self.trigger)
    self.mode = 'r' if self.outputs else ''
    self.mode += 'w' if self.to_get else ''
    assert self.mode != '', "ERROR: IOBlock is neither an input nor an output!"
    if 'w' in self.mode:
      assert self.cmd_labels, "ERROR: IOBlock has an input block but no " \
                              "cmd_labels specified!"
    if self.mode == 'rw' and self.device_name not in inandout_dict:
      raise IOError("The IOBlock has inputs and outputs but the Inout class "
                    "is not rw")
    elif self.mode == 'r' and self.device_name not in in_dict:
      raise IOError("The IOBlock has inputs but the Inout class is write-only")
    elif self.mode == 'w' and self.device_name not in out_dict:
      raise IOError("The IOBlock has outputs but the Inout class is read-only")
    self.device.open()
    if 'w' in self.mode:
      self.device.set_cmd(*self.initial_cmd)

  def read(self):
    """Will read the device and send the data."""

    if self.streamer:
      if self.stream_idle:
        self.device.start_stream()
        self.stream_idle = False
      data = self.device.get_stream()
    else:
      data = self.device.get_data()
    if isinstance(data, dict):
      pass
    elif isinstance(data[0], list):
      data[0] = [i - self.t0 for i in data[0]]
    else:
      data[0] -= self.t0
    self.send(data)

  def loop(self):
    if 'r' in self.mode:
      if self.trigger is not None:
        # To avoid useless loops if triggered input only
        if self.mode == 'r' or self.inputs[self.trigger].poll():
          self.inputs[self.trigger].recv()
          self.read()
      else:
        self.read()
    if 'w' in self.mode:
      lst = self.get_last(self.to_get)
      cmd = []
      for label in self.cmd_labels:
        cmd.append(lst[label])
      self.device.set_cmd(*cmd)

  def finish(self):
    if self.streamer:
      self.device.stop_stream()
    if self.exit_values is not None:
      self.device.set_cmd(*self.exit_values)
    self.device.close()
