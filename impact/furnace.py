# coding: utf-8

"""
Program used to control the solidification furnace.

It uses a Labjack T7 to send PWN signals to the transistors controlling the
heating element of each section of the furnace. The temperature of each section
is measured using a thermocouple.
"""

import crappy

SHOW_PID = 3

MED = 20
MEAN = 50

FREQ = 100
P = .25
I = .03
D = 20

pins = [0, 2, 3, 4, 5]

T = 200

v = dict()
v[0] = T
v[2] = T + 10
v[3] = T
v[4] = T
v[5] = T


class dc_to_clk:
  def __init__(self, lbl):
    self.lbl = lbl

  def evaluate(self, data):
    # data[self.l] = int(data[self.l] * 80000000 / FREQ)
    if data[self.lbl] < 0.01:
      cmd = 0
    elif data[self.lbl] > .99:
      cmd = 1
    else:
      cmd = data[self.lbl]
    data[self.lbl] = int((1 - cmd) * 80000000 / FREQ)
    return data


clock_config = [
                ('DIO_EF_CLOCK0_ENABLE', 0),
                ('DIO_EF_CLOCK0_DIVISOR', 1),
                ('DIO_EF_CLOCK0_ROLL_VALUE', int(80000000/FREQ)),
                ('DIO_EF_CLOCK0_ENABLE', 1),
                ]


def pwm_config(i):
  return [
      ('DIO%d_EF_ENABLE' % i, 0),
      ('DIO%d_EF_INDEX' % i, 0),
      ('DIO%d_EF_OPTIONS' % i, 0),
      ('DIO%d_EF_CONFIG_A' % i, int(80000000 / FREQ * .5)),
      ('DIO%d_EF_ENABLE' % i, 1),
    ]


# g = crappy.blocks.Generator([dict(type='constant', condition=None,
# value=200)])
pwm_chan = [dict(name="DIO%d_EF_CONFIG_A" % i,
  direction=1, write_at_open=pwm_config(i)) for i in pins]
# Adding the clock config to the first chan
pwm_chan[0]['write_at_open'][0:0] = clock_config

th_chan = [dict(name='AIN%d' % i, thermocouple='K') for i in pins]


lj = crappy.blocks.IOBlock("Labjack_t7", channels=pwm_chan+th_chan,
  labels=['t(s)']+['T%d' % i for i in pins],
  cmd_labels=['pwm%d' % i for i in pins], verbose=True)

pid_list = []
gen_list = []
graph_cmd = crappy.blocks.Grapher(*[('t(s)', 'pwm%d' % i) for i in pins])
for i in pins:
  pid_list.append(crappy.blocks.PID(P, I if i != 5 else 0, D,
    input_label='T%d' % i,
    out_max=1, out_min=0,
    i_limit=.5,
    send_terms=(SHOW_PID is not None and i == SHOW_PID),
    labels=['t(s)', 'pwm%d' % i]))

  gen_list.append(crappy.blocks.Generator(
    [dict(type='constant', condition=None, value=v[i])]))

  crappy.link(gen_list[-1], pid_list[-1])
  crappy.link(pid_list[-1], lj, modifier=dc_to_clk('pwm%d' % i))
  crappy.link(lj, pid_list[-1],
      modifier=[crappy.modifier.Median(MED), crappy.modifier.Moving_avg(MEAN)])
  crappy.link(pid_list[-1], graph_cmd)

graph = crappy.blocks.Grapher(*[('t(s)', 'T%d' % i) for i in pins])
crappy.link(lj, graph,
    modifier=[crappy.modifier.Median(MED), crappy.modifier.Moving_avg(MEAN)])


if SHOW_PID:
  graph_pid = crappy.blocks.Grapher(
      ('t(s)', 'p_term'),
      ('t(s)', 'i_term'),
      ('t(s)', 'd_term'),
      ('t(s)', 'pwm%d' % SHOW_PID))
  crappy.link(pid_list[pins.index(SHOW_PID)], graph_pid)

crappy.start()
