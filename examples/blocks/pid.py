# coding: utf-8

"""
This example demonstrates the use of the PID Block. It does not require any
specific hardware to run, but necessitates the matplotlib Python module to be
installed.

The PID Block emulates a PID controller. It takes as inputs a target value and
a measured value, and outputs the best command to set on an actuator so that
the measured values matches the target as soon as possible. Several parameters
allow to tune the behavior of the PID for a given context.

Here, the PID Block is used for controlling a Machine Block driving a
FakeDCMotor. A Generator Block generates a target speed value to reach, and
based on the target and measured speed values the PID outputs a tension
command to set on the FakeDCMotor. A Grapher allows to follow the simultaneous
variations of the target and the command speed.

After starting this script, just watch how the measured speed value evolves to
match the target speed, and how it reacts when the target changes. You can try
to play with the PID settings to see how that impacts its performance and
stability. This demo ends after 32s. You can also hit CTRL+C to stop it
earlier, but it is not a clean way to stop Crappy.
"""

import crappy

if __name__ == "__main__":

  # This Generator Block generates the target speed value that the FakeDCMotor
  # should reach. The target signal has a complex design, to demonstrate the
  # reactivity of the PID. The value is sent to the PID, that then calculates
  # the best tension to input on the FakeDCMotor. It is also sent to the
  # Grapher Block for display
  gen = crappy.blocks.Generator(
      # Generating a quite complex Path, to demonstrate the capacity of the PID
      ({'type': 'Constant', 'value': 1000, 'condition': 'delay=3'},
       {'type': 'Ramp', 'speed': 100, 'condition': 'delay=5', 'init_value': 0},
       {'type': 'Constant', 'value': 1800, 'condition': 'delay=3'},
       {'type': 'Constant', 'value': 500, 'condition': 'delay=3'},
       {'type': 'Sine', 'amplitude': 2000, 'offset': 1000, 'freq': .3,
        'condition': 'delay=15'}),
      spam=True,  # Sends a value at each loop, for a nice display on the graph
      freq=50,  # Lowering the default frequency because it's just a demo
      cmd_label='target',  # The label carrying the target speed

      # Sticking to default for the other arguments
  )

  # This Machine Block drives a single FakeDCMotor in speed. It takes as input
  # the tension command generated by the PID Block, and outputs the current
  # measured speed value for the PID to process. It also sends this value to
  # the Grapher Block for display
  mot = crappy.blocks.Machine(
      # The driven Actuator is a FakeDCMotor with the following characteristics
      ({'type': 'FakeDCMotor',
        'cmd_label': 'pid',
        'mode': 'speed',
        'speed_label': 'measured',
        'kv': 1000,
        'inertia': 4,
        'rv': .2,
        'fv': 1e-5},),
      freq=50,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This PID Block takes the measured speed value of the Machine Block as an
  # input, as well as the target speed value. It then calculates the best
  # tension command to set on the FakeDCMotor, and sends it to the Machine 
  # Block
  pid = crappy.blocks.PID(
      kp=0.038,  # The P gain of the PID
      ki=0.076,  # The I gain of the PID
      kd=0.0019,  # The D gain of the PID
      out_max=10,  # The maximum output value of the PID, here in Volts
      out_min=-10,  # The minimum output value of the PID, here in Volts
      setpoint_label='target',  # The label carrying the setpoint of the PID
      input_label='measured',  # The label carrying the measured value
      i_limit=(-5, 5),  # The upper and lower limits for the I term of the PID
      freq=50,  # Lowering the default frequency because it's just a demo

      # Sticking to default for the other arguments
  )

  # This Grapher Block displays simultaneously the target speed for the Machine
  # Block, and the actual measured speed. It allows to see how the measured
  # speed matches the target after some time, and how reactive the PID is
  graph = crappy.blocks.Grapher(
      # The names of the labels to plot on the graph
      ('t(s)', 'measured'), ('t(s)', 'target'),

      # Sticking to default for the other arguments
  )

  # Linking the Block so that the information is correctly sent and received
  # Notice how the PID and the Machine Block are linked together in a feedback
  # loop
  crappy.link(gen, pid)
  crappy.link(mot, pid)
  crappy.link(pid, mot)
  crappy.link(mot, graph)
  crappy.link(gen, graph)

  # Mandatory line for starting the test, this call is blocking
  crappy.start()
