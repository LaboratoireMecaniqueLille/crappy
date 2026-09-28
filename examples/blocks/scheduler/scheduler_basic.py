# coding: utf-8

"""
This example introduces the Scheduler Block with a simple Load -> Hold ->
Unload procedure. It uses a FakeMachine, so no hardware is required. The
Grapher needs pyqtgraph and PyQt6.

Each State supplies a constant speed command and a Delay condition selecting
the next State. The final command sets the speed to zero before the script
ends. The Grapher displays the simulated force against displacement.
"""

import crappy
from crappy.blocks.schedulers import State, Constant, Delay


if __name__ == '__main__':

  # Each State names its output and the condition for leaving it.
  scheduler = crappy.blocks.Scheduler(
    states=(State('Load', (Constant('cmd', 1.0),), ((Delay(2), 'Hold'),)),
            State('Hold', (Constant('cmd', 0.0),), ((Delay(1), 'Unload'),)),
            State('Unload', (Constant('cmd', -1.0),), ((Delay(2), 'End'),))),
    output_labels=('cmd',),
    last_output={'cmd': 0.0},
    end_delay=0.5,
    freq=50)

  # FakeMachine accepts the speed command and reports force and displacement
  machine = crappy.blocks.FakeMachine(mode='speed', cmd_label='cmd',
                                      sigma={}, freq=50)
  graph = crappy.blocks.Grapher(('x(mm)', 'F(N)'))

  crappy.link(scheduler, machine)
  crappy.link(machine, graph)

  # The procedure ends on its own, but can also be stopped early
  stop = crappy.blocks.StopButton()
  crappy.start()
