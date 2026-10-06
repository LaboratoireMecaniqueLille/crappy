# coding: utf-8

"""
This example uses ordinary Python functions as Scheduler outputs and stop
conditions. It runs with a FakeMachine and requires no hardware. The stop
button requires PyQt6.

The speed drops when measured force rises. A custom condition combines force
and displacement to select Hold. A five-second timeout provides a finite
fallback if the target is not reached.
"""

import crappy
from crappy.blocks.schedulers import State, Constant, Delay


def adaptive_speed(_dt, data):
  """Reduce loading speed after the measured force reaches 20 kN."""

  force_values = data.get('F(N)')
  force = force_values[-1] if force_values else 0
  return {'cmd': 0.8 if force < 20000 else 0.25}


def target_reached(_dt, data):
  """Require both force and displacement to pass their targets."""

  force_values = data.get('F(N)')
  position_values = data.get('x(mm)')
  return (bool(force_values) and bool(position_values) and
          force_values[-1] > 25000 and position_values[-1] > 0.6)


if __name__ == '__main__':

  scheduler = crappy.blocks.Scheduler(
    states=(State('Load', (adaptive_speed,),
                  ((target_reached, 'Hold'), (Delay(5), 'End'))),
            State('Hold', (Constant('cmd', 0.0),), ((Delay(1), 'End'),))),
    output_labels=('cmd',),
    input_labels=('F(N)', 'x(mm)'),
    last_output={'cmd': 0.0},
    end_delay=0.5,
    freq=50)

  machine = crappy.blocks.FakeMachine(mode='speed', cmd_label='cmd',
                                      sigma={}, freq=50)
  reader = crappy.blocks.LinkReader(name='Scheduler command')

  crappy.link(scheduler, machine)
  crappy.link(machine, scheduler)
  crappy.link(scheduler, reader)

  stop = crappy.blocks.StopButton()
  crappy.start()
