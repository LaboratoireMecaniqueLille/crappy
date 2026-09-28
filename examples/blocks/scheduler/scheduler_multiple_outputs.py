# coding: utf-8

"""
This example uses one Scheduler to drive two FakeDCMotor Actuators through a
Machine Block. No hardware is required. The Grapher needs pyqtgraph and PyQt6.

The Scheduler starts both voltages at zero. Each State changes only one
voltage: the other retains its last value in the Scheduler's output buffer.
The motors are stopped together when the procedure reaches End.
"""

import crappy
from crappy.blocks.schedulers import State, Constant, Delay


if __name__ == '__main__':

  scheduler = crappy.blocks.Scheduler(
    states=(State('First motor', (Constant('motor_1(V)', 4.0),),
                  ((Delay(3), 'Both motors'),)),
            # motor_1 stays at 4 V because this State only updates motor_2
            State('Both motors', (Constant('motor_2(V)', -4.0),),
                  ((Delay(3), 'Reverse first'),)),
            # motor_2(V) stays at -4 V until End resets both commands
            State('Reverse first', (Constant('motor_1(V)', -4.0),),
                  ((Delay(3), 'End'),))),
    output_labels=('motor_1(V)', 'motor_2(V)'),
    init_values={'motor_1(V)': 0.0, 'motor_2(V)': 0.0},
    last_output={'motor_1(V)': 0.0, 'motor_2(V)': 0.0},
    end_delay=0.5,
    freq=50)

  # Each motor listens to a different voltage and reports its own speed
  machine = crappy.blocks.Machine((
    {'type': 'FakeDCMotor', 'mode': 'speed',
     'cmd_label': 'motor_1(V)', 'speed_label': 'motor_1(RPM)', 'kv': 500},
    {'type': 'FakeDCMotor', 'mode': 'speed',
     'cmd_label': 'motor_2(V)', 'speed_label': 'motor_2(RPM)', 'kv': 500}),
    freq=50)
  graph = crappy.blocks.Grapher(('t(s)', 'motor_1(RPM)'),
                                ('t(s)', 'motor_2(RPM)'))

  crappy.link(scheduler, machine)
  crappy.link(machine, graph)

  stop = crappy.blocks.StopButton()
  crappy.start()
