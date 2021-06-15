# coding: utf-8

from time import time

from .path import Path


class Cyclic_ramp(Path):
  """A "boosted" :ref:`ramp` path: will take TWO values and conditions.

  Note:
    It will make a ramp of speed ``speed1``, switch to the second when the
    first condition is reached and return to the first when the second
    condition is reached.

    This will be done ``cycles`` times (supporting half cycles for ending after
    the first condition)
  """

  def __init__(self, time, cmd, condition1, condition2, speed1, speed2,
               cycles=1, verbose=False):
    """Sets the args and initializes parent class.

    Args:
      time:
      cmd:
      condition1 (:obj:`str`): Representing the condition to switch to
        ``speed2``. See :ref:`generator path` for more info.
      condition2 (:obj:`str`): Representing the condition to switch to
        ``speed1``. See :ref:`generator path` for more info.
      speed1: Speed of the first ramp.
      speed2: Speed of the second ramp.
      cycles: Number of time we should be doing this.

        Note:
          ``cycles = 0`` will make it loop forever.

      verbose:

    Note:
      ::

        [{'type': 'cyclic_ramp', 'speed1': 5, 'condition1': 'AIN0>2',
        'speed2': -2, 'condition2': 'AIN1<1', 'cycles': 5}]

      is equivalent to
      ::

        [{'type': 'ramp', 'speed': 5,'condition': 'AIN0>2'},
        {'type': 'ramp', 'value': -2, 'condition': 'AIN1<1'}] * 5
    """

    Path.__init__(self, time, cmd)
    self.speed = (speed1, speed2)
    self.condition1 = self.parse_condition(condition1)
    self.condition2 = self.parse_condition(condition2)
    self.cycles = int(2 * cycles)  # Logic in this class will be in half-cycle
    self.cycle = 0
    self.verbose = verbose

  def get_cmd(self, data):
    if 0 < self.cycles <= self.cycle:
      raise StopIteration
    if not self.cycle % 2 and self.condition1(data) or\
          self.cycle % 2 and self.condition2(data):
      t = time()
      self.cmd += self.speed[self.cycle % 2] * (t - self.t0)
      self.t0 = t
      self.cycle += 1
      if self.verbose:
        print("cyclic ramp {}/{}".format(self.cycle, self.cycles))
    return self.speed[self.cycle % 2] * (time() - self.t0) + self.cmd
