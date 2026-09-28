# coding: utf-8

"""
This example branches according to FakeMachine feedback. No hardware is
required. The Grapher needs pyqtgraph and PyQt6.

Load normally reaches the force target and moves to Hold. If it has not
reached the target after two seconds, the later timeout condition selects
Abort instead. Increase the force target to try that branch.
"""

import crappy
from crappy.blocks.schedulers import State, Constant, Delay, Compare


if __name__ == '__main__':

  scheduler = crappy.blocks.Scheduler(
    states=(State('Load', (Constant('cmd', 1.0),), (
        # Conditions are ordered: force wins if both become true together
        (Compare('F(N)', '>', value=30000, mode='last'), 'Hold'),
        (Delay(2), 'Abort'))),
            State('Hold', (Constant('cmd', 0.0),), ((Delay(1), 'Unload'),)),
            State('Unload', (Constant('cmd', -1.0),), ((Delay(2), 'End'),)),
            State('Abort', (Constant('cmd', 0.0),), ((Delay(0.5), 'End'),))),
    output_labels=('cmd',),
    input_labels=('F(N)',),
    last_output={'cmd': 0.0},
    end_delay=0.5,
    freq=50)

  machine = crappy.blocks.FakeMachine(mode='speed', cmd_label='cmd',
                                      sigma={}, freq=50)
  graph = crappy.blocks.Grapher(('t(s)', 'F(N)'))

  # Feedback lets the Scheduler select Hold or Abort from measured force
  crappy.link(scheduler, machine)
  crappy.link(machine, scheduler)
  crappy.link(machine, graph)

  stop = crappy.blocks.StopButton()
  crappy.start()
