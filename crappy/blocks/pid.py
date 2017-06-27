#coding: utf-8
from __future__ import division

from time import time
from .masterblock import MasterBlock

class PID(MasterBlock):
  """
  This PID class will continuously adjust the output based on the target
  and the actual value
  """
  def __init__(self,kp,ki=0,kd=0,**kwargs):
    MasterBlock.__init__(self)
    self.niceness = -10
    for arg,default in [('freq',200),
                    ('out_max',None),
                    ('out_min',None),
                    ('target_label','cmd'),
                    ('input_label','V'),
                    ('time_label','t(s)'),
                    ('labels',['t(s)','pid']),
                    ('reverse',False),
                    ('send_terms',False) # For debug, mostly
                    ]:
      setattr(self,arg,kwargs.pop(arg,default))
    assert not kwargs,"PID got incorrect kwarg(s): "+str(kwargs)
    self.set_k(kp,ki,kd)
    self.i_term = 0
    self.last_val = 0
    if self.send_terms:
      self.labels.extend(['p_term','i_term','d_term'])

  def clamp(self,v):
    return max(v if self.out_max is None else min(v,self.out_max),self.out_min)

  def set_k(self,kp,ki=0,kd=0):
    s = -1 if self.reverse else 1
    self.kp = s*kp
    self.ki = s*ki/self.freq # We are working at constant loop rate
    self.kd = s*kd*self.freq # so this will save a bit of comutation

  def loop(self):
    data = self.get_last()
    #t = time()
    t = data[self.time_label]
    target = data[self.target_label]
    feedback = data[self.input_label]
    diff = target-feedback

    p_term = self.kp*diff

    # Classical approach
    #d_term = self.kd*(diff - self.last_val)
    #self.last_val = diff
    # Alternative: ignore setpoint to avoid derivative kick
    d_term = -self.kd*(feedback-self.last_val)
    self.last_val = feedback

    self.i_term += self.ki*diff
    out = p_term+self.i_term+d_term
    if not self.out_min < out < self.out_max:
      out = self.clamp(out)
      self.i_term = self.clamp(self.i_term)
    # So that I doesn't keep integrating too far when reaching limit values
    #if out > self.out_max:
    #  self.i_term = min(self.i_term,self.out_max - p_term - d_term)
    #  out = self.out_max
    #elif out < self.out_min:
    #  self.i_term = max(self.i_term,self.out_min - p_term - d_term)
    #  out = self.out_min
    if self.send_terms:
      self.send([t,out,p_term,self.i_term,d_term])
    else:
      self.send([t,out])
