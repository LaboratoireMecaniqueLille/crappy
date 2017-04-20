#coding: utf-8
from __future__ import print_function,division

from time import time,sleep

from .masterblock import MasterBlock
from . import generator_path
from .._global import CrappyStop

class Generator(MasterBlock):
  def __init__(self,path=[],**kwargs):
    MasterBlock.__init__(self)
    for arg,default in [('freq',100),
                        ('labels',['t(s)','cmd']),
                        ('cmd',0), # First value
                        ('repeat',False), # Start over when done ?
                       ]:
      setattr(self,arg,kwargs.get(arg,default))
    self.path = path
    assert all([hasattr(generator_path,d['type']) for d in self.path]),\
        "Invalid path in signal generator:"\
        +str(filter(lambda s: not hasattr(generator_path,s['type']),self.path))

  def prepare(self):
    self.path_id = -1 # Will be incremented to 0 on first next_path
    self.last_t = time()
    self.last_data = {}
    self.next_path()

  def next_path(self):
    self.path_id += 1
    if self.path_id >= len(self.path):
      if self.repeat:
        self.path_id = 0
      else:
        print("Signal generator terminated!")
        MasterBlock.stop_all()
        raise CrappyStop("Signal Generator terminated")
    print("[Signal Generator] Next step({}):".format(self.path_id),
        self.path[self.path_id])
    kwargs = {'cmd':self.cmd, 'time':self.last_t}
    kwargs.update(self.path[self.path_id])
    del kwargs['type']
    name = self.path[self.path_id]['type'].capitalize()
    # Instanciating the new path class for the next step
    self.current_path = getattr(generator_path,name)(**kwargs)

  def loop(self):
    data = self.get_all_last()
    try:
      cmd = self.current_path.get_cmd(data)
    except StopIteration:
      self.next_path()
      self.loop()
      return
    if cmd is not None:
      self.cmd = cmd
    self.send([time(),self.cmd])
