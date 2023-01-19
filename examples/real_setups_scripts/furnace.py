# coding: utf-8

"""
Program used to control the solidification furnace.

It uses a Labjack T7 to send PWN signals to the transistors controlling the
heating element of each section of the furnace. The temperature of each section
is measured using a thermocouple.
"""

import crappy

# The parameters of the PID
P = .25
I = .03
D = 20

# The frequency of the PWM
FREQ = 100
# Parameters for averaging the signals
MED = 20
MEAN = 50

# The pins to drive
PINS = [0, 2, 3, 4, 5]
# For which pin the PID value breakout should be displayed, or False
SHOW_PID = 3

# The target temperature for the different pins
TEMP = {0: 200, 2: 210, 3: 200, 4: 200, 5: 200}

# Configuration commands for the clock of the Labjack
clock_config = [('DIO_EF_CLOCK0_ENABLE', 0),
                ('DIO_EF_CLOCK0_DIVISOR', 1),
                ('DIO_EF_CLOCK0_ROLL_VALUE', int(80000000 / FREQ)),
                ('DIO_EF_CLOCK0_ENABLE', 1)]


class dc_to_clk(crappy.Modifier):
  """This class converts a DC command to a duration in clock cycles."""

  def __init__(self, label):
    """Initializes the parent class and sets the argument."""

    super().__init__()
    self._label = label

  def evaluate(self, data):
    """Converts the value received on the given label from a DC voltage to a
    number of clock cycles."""

    if data[self._label] < 0.01:
      cmd = 0
    elif data[self._label] > .99:
      cmd = 1
    else:
      cmd = data[self._label]

    data[self._label] = int((1 - cmd) * 80000000 / FREQ)
    return data


def pwm_config(i):
  """Generates startup commands for configuring the ith pin of the Labjack."""

  return [(f'DIO{i}_EF_ENABLE', 0),
          (f'DIO{i}_EF_INDEX', 0),
          (f'DIO{i}_EF_OPTIONS', 0),
          (f'DIO{i}_EF_CONFIG_A', int(80000000 / FREQ * .5)),
          (f'DIO{i}_EF_ENABLE', 1)]


if __name__ == '__main__':

  # Generating the arguments for the Labjack channels driving the PWMs
  pwm_chan = [dict(name=f"DIO{i}_EF_CONFIG_A", direction=1,
                   write_at_open=pwm_config(i)) for i in PINS]
  # Adding the clock configuration to the first channel
  pwm_chan[0]['write_at_open'][0:0] = clock_config

  # Generating the arguments for the Labjack channels reading the thermocouples
  th_chan = [dict(name=f'AIN{i}', thermocouple='K') for i in PINS]

  # The Block communicating with the Labjack
  labjack = crappy.blocks.IOBlock("LabjackT7",
                                  channels=pwm_chan + th_chan,
                                  labels=['t(s)'] + [f'T{i}' for i in PINS],
                                  cmd_labels=[f'pwm{i}' for i in PINS],
                                  verbose=True)

  # the Blocks displaying the temperature and command values in real-time
  graph_cmd = crappy.blocks.Grapher(*[('t(s)', f'pwm{i}') for i in PINS])
  graph_temp = crappy.blocks.Grapher(*[('t(s)', f'T{i}') for i in PINS])
  crappy.link(labjack, graph_temp,
              modifier=[crappy.modifier.Median(MED),
                        crappy.modifier.MovingAvg(MEAN)])

  pid_list = []
  gen_list = []

  # For each pin, instantiating a PID and a Generator for driving it
  for i in PINS:
    pid_list.append(crappy.blocks.PID(P, I if i != 5 else 0, D,
                                      input_label=f'T{i}',
                                      out_max=1, out_min=0,
                                      i_limit=(0.5, -0.5),
                                      send_terms=(SHOW_PID is not None
                                                  and i == SHOW_PID),
                                      labels=['t(s)', f'pwm{i}']))

    gen_list.append(crappy.blocks.Generator(
      [dict(type='Constant', condition=None, value=TEMP[i])]))

    crappy.link(gen_list[-1], pid_list[-1])
    crappy.link(pid_list[-1], labjack, modifier=[dc_to_clk(f'pwm{i}')])
    crappy.link(labjack, pid_list[-1],
                modifier=[crappy.modifier.Median(MED),
                          crappy.modifier.MovingAvg(MEAN)])
    crappy.link(pid_list[-1], graph_cmd)

  # If requested to display details about a given PID, creating a Grapher
  if SHOW_PID:
    graph_pid = crappy.blocks.Grapher(('t(s)', 'p_term'),
                                      ('t(s)', 'i_term'),
                                      ('t(s)', 'd_term'),
                                      ('t(s)', f'pwm{SHOW_PID}'))
    crappy.link(pid_list[PINS.index(SHOW_PID)], graph_pid)

  # Starting the test
  crappy.start()
