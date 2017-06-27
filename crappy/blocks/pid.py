#coding: utf-8
from __future__ import division

from .masterblock import MasterBlock

class PID(MasterBlock):
  """
  This PID class will continuously adjust the output based on the target
  and the actual value
  """
  def __init__(self,kp,ki=0,kd=0,**kwargs):
    MasterBlock.__init__(self)
    self.niceness = -10
    for arg,default in [('freq',500),
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

  def begin(self):
    self.last_t = self.t0
    data = [l.recv_last(True) for l in self.inputs]
    for i,r in enumerate(data):
      if self.target_label in r:
        self.target_link_id = i
      if self.input_label in r and self.time_label in r:
        self.feedback_link_id = i
    assert hasattr(self,"target_link_id"),"[PID] Error: no link containing"\
        " target label {}".format(self.target_label)
    assert hasattr(self,"feedback_link_id"),"[PID] Error: no link containing"\
        " input label {} and time label {}".format(
        self.input_label,self.time_label)
    assert set(range(len(self.inputs))) ==\
        set((self.target_link_id,self.feedback_link_id)),"[PID] Error: useless"\
        " link(s)! Make sure PID block does not have extra inputs"
    self.last_target = data[self.target_link_id][self.target_label]
    self.last_t = data[self.feedback_link_id][self.time_label]
    # For the classical D approach:
    #self.last_val = self.last_target -\
    #    data[self.feedback_link_id][self.input_label]
    # When ignore setpoint mode
    self.last_val = data[self.feedback_link_id][self.input_label]


    if self.send_terms:
      self.send([self.last_t,0,0,0,0])
    else:
      self.send([self.last_t,0])

  def clamp(self,v):
    return max(v if self.out_max is None else min(v,self.out_max),self.out_min)

  def set_k(self,kp,ki=0,kd=0):
    s = -1 if self.reverse else 1
    self.kp = s*kp
    self.ki = s*ki
    self.kd = s*kd

  def loop(self):
    data = self.inputs[self.feedback_link_id].recv_last(True)
    t = data[self.time_label]
    dt = t - self.last_t
    feedback = data[self.input_label]
    if self.feedback_link_id == self.target_link_id:
      target = data[self.target_label]
    else:
      data = self.inputs[self.target_link_id].recv_last()
      if data is None:
        target = self.last_target
      else:
        target = data[self.target_label]
        self.last_target = target
    diff = target-feedback

    p_term = self.kp*diff
    self.last_t = t

    # Classical approach
    #d_term = self.kd*(diff - self.last_val)
    #self.last_val = diff
    # Alternative: ignore setpoint to avoid derivative kick
    d_term = -self.kd*(feedback-self.last_val)/dt
    self.last_val = feedback

    self.i_term += self.ki*diff*dt
    out = p_term+self.i_term+d_term
    if not self.out_min < out < self.out_max:
      out = self.clamp(out)
      self.i_term = self.clamp(self.i_term)
    if self.send_terms:
      self.send([t,out,p_term,self.i_term,d_term])
    else:
      self.send([t,out])
