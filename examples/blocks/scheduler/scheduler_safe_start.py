# coding: utf-8

"""
This example shows what ``safe_start`` actually waits for. It uses a Button,
a FakeMachine and a Grapher, but no physical hardware. The Grapher needs
pyqtgraph and PyQt6.

The Button does not publish its ``step`` label until clicked. FakeMachine
publishes its initial ``F(N)`` measurement at startup. With both labels listed
in ``input_labels`` and ``safe_start=True``, the Scheduler evaluates no State
output function until it has observed both values. Stop conditions are still
checked while waiting, so the click can move Wait to Load as soon as it is
received. Load's own two-second timer starts on State entry.
"""

import crappy
from crappy.blocks.schedulers import State, Constant, Delay, Compare


if __name__ == '__main__':

  scheduler = crappy.blocks.Scheduler(
    states=(State('Wait', (Constant('cmd', 0.0),),
                  ((Compare('step', '>', value=0, mode='last'), 'Load'),)),
            State('Load', (Constant('cmd', 1.0),), ((Delay(2), 'End'),))),
    output_labels=('cmd',),
    input_labels=('step', 'F(N)'),
    safe_start=True,
    last_output={'cmd': 0.0},
    end_delay=0.5,
    freq=50)

  # With send_0=False, step is absent until the user clicks the button
  button = crappy.blocks.Button(send_0=False, label='step', freq=30)
  machine = crappy.blocks.FakeMachine(mode='speed', cmd_label='cmd',
                                      sigma={}, freq=50)
  graph = crappy.blocks.Grapher(('t(s)', 'x(mm)'))

  crappy.link(button, scheduler)
  crappy.link(machine, scheduler)
  crappy.link(scheduler, machine)
  crappy.link(machine, graph)

  stop = crappy.blocks.StopButton()
  crappy.start()
