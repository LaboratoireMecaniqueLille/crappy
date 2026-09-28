# coding: utf-8

"""
This example uses a cyclic Scheduler graph to move a FakeMachine repeatedly
between two displacement thresholds. No hardware is required. The Grapher
needs pyqtgraph and PyQt6.

Each State is re-entered on every cycle. Its Crossing condition is reset on
entry, so it waits for a new threshold crossing. Use the StopButton to end
the test.
"""

import crappy
from crappy.blocks.schedulers import State, Constant, Crossing


if __name__ == '__main__':

  scheduler = crappy.blocks.Scheduler(
    states=(State('Loading', (Constant('cmd', 1.0),),
                  ((Crossing('x(mm)', 0.8, 'rising'), 'Unloading'),)),
            State('Unloading', (Constant('cmd', -1.0),),
                  ((Crossing('x(mm)', 0.2, 'falling'), 'Loading'),))),
    output_labels=('cmd',),
    input_labels=('x(mm)',),
    freq=50)

  # Disable measurement noise so each crossing is easy to see in the plot
  machine = crappy.blocks.FakeMachine(mode='speed', cmd_label='cmd',
                                      sigma={}, freq=50)
  graph = crappy.blocks.Grapher(('t(s)', 'x(mm)'))

  crappy.link(scheduler, machine)
  crappy.link(machine, scheduler)
  crappy.link(machine, graph)

  # The State graph intentionally has no End transition
  stop = crappy.blocks.StopButton()
  crappy.start()
