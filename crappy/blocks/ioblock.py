#coding: utf-8
from __future__ import print_function

from .masterblock import MasterBlock
from ..inout import inout_list

class IOBlock(MasterBlock):
  def __init__(self,name,**kwargs):
    MasterBlock.__init__(self)
    for arg,default in [('freq',None),
                        ('verbose',False),
                        ('labels',['t(s)','1']),
                        ('cmd_labels',['cmd']),
                        ('trigger',None)
                        ]:
      if arg in kwargs:
        setattr(self,arg,kwargs[arg])
        del kwargs[arg]
      else:
        setattr(self,arg,default)
    self.device_name = name
    self.device_kwargs = kwargs

  def prepare(self):
    self.to_get = range(len(self.inputs))
    if self.trigger is not None:
      self.to_get.remove(self.trigger)

    self.device = inout_list[self.device_name](**self.device_kwargs)
    self.device.open()

  def loop(self):
    if self.trigger is None or self.inputs[self.trigger].poll():
      if self.trigger is not None:
        self.inputs[self.trigger].recv()
      data = self.device.get_data()
      data[0] -= self.t0
      self.send(data)
    l = self.get_last(self.to_get)
    cmd = []
    for label in self.cmd_labels:
      cmd.append(l[label])
    self.device.set_cmd(*cmd)
