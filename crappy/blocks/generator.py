#coding: utf-8
from __future__ import print_function,division

from time import time

from .masterblock import MasterBlock
from . import generator_path
from .._global import CrappyStop

class Generator(MasterBlock):
  """
  This block is used to generate a signal. It can be used to drive a machine.
  This block can take inputs, and each path can use these inputs to
  take decisions.
  Args:
  path:
    It must be a list of dict, each dict providing the parameters to generate
    the path
    This dict MUST have a key 'type'
    The Generator will then instanciate generator_path.**type** with all the
    other keys as kwargs, adding the current cmd and the time.
    On each round, it will call get_cmd method of this class, passing data
    until it raise StopIteration. It will then skip to the next path.
  When all paths are over, it will stop Crappy by raising CrappyStop
  unless repeat is set to True. If so, it will start over indefinelty.
  freq: The frequency of the block
    if set and positive, the generator will try to send the command at this
    frequency (Hz). Else, it will go as fast as possible
    It relies on the MasterBlock freq control scheme (see masterblock.py)
  cmd_label: the label of the command to send in the links
    default: 'cmd'
  cmd: The first value of the command
    default: 0
    Some paths may rely on the previous value to guarantee a continuous signal
    This argument sets the initial value for the first signal
  repeat: Loop over the paths or stop when done ?
    If False, the block will raise a CrappyStop exception to end the program
    when all the paths have been executed
    If True, the Generator will start over and over again
  """
  def __init__(self,path=[],**kwargs):
    MasterBlock.__init__(self)
    for arg,default in [('freq',None),
                        ('cmd_label','cmd'),
                        ('cycle_label','cycle'),
                        ('cmd',0), # First value
                        ('repeat',False), # Start over when done ?
                       ]:
      if arg in kwargs:
        setattr(self,arg,kwargs[arg])
        del kwargs[arg]
      else:
        setattr(self,arg,default)
    assert not kwargs,"generator: unknow kwargs: "+str(kwargs)
    self.path = path
    assert all([hasattr(generator_path,d['type']) for d in self.path]),\
        "Invalid path in signal generator:"\
        +str(filter(lambda s: not hasattr(generator_path,s['type']),self.path))
    self.labels = ['t(s)',self.cmd_label,self.cycle_label]

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

  def begin(self):
    self.send([self.last_t-self.t0,self.cmd,self.path_id])
    self.current_path.t0 = self.t0

  def loop(self):
    data = self.get_all_last()
    data[self.cmd_label] = [self.cmd] # Add my own cmd to the dict
    try:
      cmd = self.current_path.get_cmd(data)
    except StopIteration:
      self.next_path()
      self.loop()
      return
    if cmd is not None: # If next_path returns None, do not update cmd
      self.cmd = cmd
    self.last_t = time()
    self.send([self.last_t-self.t0,self.cmd,self.path_id])
