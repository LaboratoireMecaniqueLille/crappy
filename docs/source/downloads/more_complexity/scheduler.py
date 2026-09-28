# coding: utf-8

# [scheduler-start]
import crappy
from crappy.blocks.schedulers import State, Constant, Delay, Compare


def main() -> None:
  scheduler = crappy.blocks.Scheduler(
      states=(State('Load', (Constant('input_speed', 1.0),),
                    ((Compare('F(N)', '>', value=25_000, mode='last'), 'Hold'),
                     (Delay(2), 'Abort'))),
              State('Hold', (Constant('input_speed', 0.0),),
                    ((Delay(0.5), 'Unload'),)),
              State('Unload', (Constant('input_speed', -1.0),),
                    ((Delay(0.7), 'End'),)),
              State('Abort', (Constant('input_speed', 0.0),),
                    ((Delay(0.2), 'End'),))),
      output_labels=('input_speed',),
      input_labels=('F(N)',),
      safe_start=True,
      last_output={'input_speed': 0.0},
      end_delay=0.2,
      freq=20)

  machine = crappy.blocks.FakeMachine(
      mode='speed', cmd_label='input_speed', sigma={}, freq=20)
  reader = crappy.blocks.LinkReader(name='Scheduler and machine', freq=20)

  crappy.link(scheduler, machine)
  crappy.link(machine, scheduler)
  crappy.link(scheduler, reader)
  crappy.link(machine, reader)

  crappy.start()


if __name__ == '__main__':
  main()
# [scheduler-end]
