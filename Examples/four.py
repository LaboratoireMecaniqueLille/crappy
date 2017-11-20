#coding: utf-8

import crappy

FREQ = 100
P = .1
I = 0
D = 0

pins= [0,2,3,4,5]

v = {}
v[0] = 200
v[2] = 200
v[3] = 200
v[4] = 200
v[5] = 200

class dc_to_clk:
  def __init__(self,lbl):
    self.l = lbl
  def evaluate(self,data):
    #data[self.l] = int(data[self.l]*80000000/FREQ)
    data[self.l] = int((1-data[self.l])*80000000/FREQ)
    return data

clock_config = [
('DIO_EF_CLOCK0_ENABLE',0),
('DIO_EF_CLOCK0_DIVISOR',1),
('DIO_EF_CLOCK0_ROLL_VALUE',int(80000000/FREQ)),
('DIO_EF_CLOCK0_ENABLE',1),
]

def pwm_config(i):
  return [
      ('DIO%d_EF_ENABLE'%i,0),
      ('DIO%d_EF_INDEX'%i,0),
      ('DIO%d_EF_OPTIONS'%i,0),
      ('DIO%d_EF_CONFIG_A'%i,int(80000000/FREQ*.5)),
      ('DIO%d_EF_ENABLE'%i,1),
      ]

#g = crappy.blocks.Generator([dict(type='constant',condition=None,value=200)])

pwm_chan = [dict(name="DIO%d_EF_CONFIG_A"%i,
  direction=1,write_at_open=pwm_config(i)) for i in pins]
# Adding the clock config to the first chan
pwm_chan[0]['write_at_open'][0:0] = clock_config

th_chan = [dict(name='AIN%d'%i,thermocouple='K') for i in pins]


lj = crappy.blocks.IOBlock("Labjack_t7",channels=pwm_chan+th_chan,
  labels=['t(s)']+['T%d'%i for i in pins],
  cmd_labels=['pwm%d'%i for i in pins])

pid_list = []
gen_list = []
graph_cmd = crappy.blocks.Grapher(*[('t(s)','pwm%d'%i) for i in pins])
for i in pins:
  pid_list.append(crappy.blocks.PID(P,I,D,
    input_label='T%d'%i,
    out_max=1,out_min=0,
    labels=['t(s)','pwm%d'%i]))

  gen_list.append( crappy.blocks.Generator(
    [dict(type='constant',condition=None,value=v[i])]))

  crappy.link(gen_list[-1],pid_list[-1])
  crappy.link(pid_list[-1],lj,condition=dc_to_clk('pwm%d'%i))
  crappy.link(lj,pid_list[-1])
  crappy.link(pid_list[-1],graph_cmd)

graph = crappy.blocks.Grapher(*[('t(s)','T%d'%i) for i in pins])
crappy.link(lj,graph)


crappy.start()
