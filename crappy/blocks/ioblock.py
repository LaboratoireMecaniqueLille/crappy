# coding: utf-8
from __future__ import print_function
from .masterblock import MasterBlock
from ..inout import inout_list, in_list, out_list


class IOBlock(MasterBlock):
  """
  This block is used to communicate with inout objects

  Then can be used as sensor, actuators or both.
  It only takes a single argument:
    name (str): The name of the inout class to instanciate
  It can take all the settings as kwargs:
    - freq (float or None): The looping frequency (see masterblock)
      set to None (default) to go as fast as possible
    - verbose (bool): Will print extra information
    - labels (list): The list of the output labels (see masterblock)
        NOTE: the first label is the time.
      default: ['t(s)','1']
    - cmd_label (list): The list of the labels carrying values for the output
      the block will call ioobject.set_cmd(...) with these values unless
      it is empty (default).
    - trigger (int or None): If the block is trigged by another block, this
      must specify the index of the input considered as a trigger.
      If set to None (default), it will run at freq if possible.
      Note: The data going through the trig link is discarded.
      Add another link if necessary
    - streamer (bool): If False (default), will call get_data
      else, will call get_stream
  """

  def __init__(self, name, **kwargs):
    MasterBlock.__init__(self)
    self.niceness = -10
    for arg, default in [('freq', None),
                         ('verbose', False),
                         ('labels', None),
                         ('cmd_labels', []),
                         ('trigger', None),
                         ('streamer', False),
                         ('initial_cmd', 0)
                         ]:
      setattr(self, arg, kwargs.pop(arg, default))

    if self.labels is None:
      if self.streamer:
        self.labels = ['t(s)', 'stream']
      else:
        self.labels = ['t(s)'] + [str(c) for c in kwargs.get("channels", ['1'])]
    self.device_name = name.capitalize()
    self.device_kwargs = kwargs
    self.stream_idle = True
    if not isinstance(self.initial_cmd, list):
      self.initial_cmd = [self.initial_cmd] * len(self.cmd_labels)

  def prepare(self):
    self.to_get = range(len(self.inputs))
    if self.trigger is not None:
      self.to_get.remove(self.trigger)
    self.mode = 'r' if self.outputs else ''
    self.mode += 'w' if self.to_get else ''
    assert self.mode != '', "ERROR: IOBlock is neither an input nor an output!"
    if 'w' in self.mode:
      assert self.cmd_labels, "ERROR: IOBlock has an input block but no" \
                              "cmd_labels specified!"
    if self.mode == 'rw':
      self.device = inout_list[self.device_name](**self.device_kwargs)
    elif self.mode == 'r':
      self.device = in_list[self.device_name](**self.device_kwargs)
    elif self.mode == 'w':
      self.device = out_list[self.device_name](**self.device_kwargs)
    self.device.open()
    if 'w' in self.mode:
      self.device.set_cmd(*self.initial_cmd)

  def read(self):
    """Will read the device and send the data"""
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
      l = self.get_last(self.to_get)
      cmd = []
      for label in self.cmd_labels:
        cmd.append(l[label])
      self.device.set_cmd(*cmd)

  def finish(self):
    if self.streamer:
      self.device.stop_stream()
    self.device.close()
