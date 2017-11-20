#coding: utf-8
from __future__ import print_function,division
import crappy

SPEED = 10

P = .1
I = .1*SPEED
D = 10/SPEED


class Delay:
  def __init__(self,delay):
    self.delay = delay
    self.t = 't(s)'
    self.v = 'pid'
    self.hist = []

  def evaluate(self,data):
    self.hist.append(data)
    r = dict(data)
    while self.hist and self.hist[0][self.t] + self.delay <= r[self.t]:
      del self.hist[0]
    v_table = [d[self.v] for d in self.hist]
    r[self.v] = sum(v_table)/len(v_table)
    return r

g = crappy.blocks.Generator([
  #dict(type='constant',condition="delay=30",value=200),
  #dict(type='constant',condition="delay=20",value=300),
  #dict(type='constant',condition="delay=20",value=400),
  dict(type='constant',condition=None,value=500)
  ])

four = crappy.blocks.Machine([dict(
  type='Fake_motor',
  cmd='pid',
  sim_speed=SPEED,
  mode='speed',
  speed_label='T',
  kv=1000,
  inertia=500,
  rv=.01,
  torque=-18,
  #initial_speed=17.8218,
  initial_speed=175,
  fv=1e-5
  )])

pid = crappy.blocks.PID(P,I,D,input_label='T',out_max=1,out_min=0,
    i_limit=.5,send_terms=True)

crappy.link(g,pid)
crappy.link(pid,four,condition=Delay(20/SPEED))
crappy.link(four,pid)

graph = crappy.blocks.Grapher(('t(s)','T'))
crappy.link(four,graph)

graph_pid = crappy.blocks.Grapher(
    ('t(s)','p_term'),
    ('t(s)','i_term'),
    ('t(s)','d_term'),
    ('t(s)','pid'))
crappy.link(pid,graph_pid)


crappy.start()
