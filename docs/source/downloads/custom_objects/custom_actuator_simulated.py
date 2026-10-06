# coding: utf-8

# [custom-actuator-start]
from time import monotonic

import crappy


# [custom-actuator-class-start]
class SimulatedStage(crappy.actuator.Actuator):

  def __init__(self, initial_position: float = 0) -> None:
    super().__init__()
    self._position = initial_position
    self._speed = 0.0
    self._last_update = None

  def open(self) -> None:
    self._last_update = monotonic()

  def _update_position(self) -> None:
    now = monotonic()
    self._position += self._speed * (now - self._last_update)
    self._last_update = now

  def set_speed(self, speed: float) -> None:
    self._update_position()
    self._speed = speed

  def get_speed(self) -> float:
    return self._speed

  def get_position(self) -> float:
    self._update_position()
    return self._position

  def stop(self) -> None:
    self._update_position()
    self._speed = 0.0

  def close(self) -> None:
    self._speed = 0.0
    self._last_update = None
# [custom-actuator-class-end]


def main() -> None:
  command = crappy.blocks.Generator(
      path=({'type': 'Cyclic',
             'value1': 2,
             'condition1': 'delay=1',
             'value2': -2,
             'condition2': 'delay=1',
             'cycles': 2},),
      cmd_label='target_speed(mm/s)',
      freq=10,
      end_delay=0.2)

  # [custom-actuator-use-start]
  stage = crappy.blocks.Machine(
      ({'type': 'SimulatedStage',
        'mode': 'speed',
        'cmd_label': 'target_speed(mm/s)',
        'speed_label': 'speed(mm/s)',
        'position_label': 'position(mm)',
        'initial_position': 10},),
      freq=10)
  # [custom-actuator-use-end]

  reader = crappy.blocks.LinkReader(name='Stage state', freq=10)

  crappy.link(command, stage)
  crappy.link(stage, reader)

  crappy.start()


if __name__ == '__main__':
  main()
# [custom-actuator-end]
