# coding: utf-8

from time import time

from .path import Path


class Cyclic(Path):
  """A "boosted" :ref:`constant` path: will take TWO values and conditions.

  Note:
    It will set the first value, switch to the second when the first condition
    is reached and return to the first when the second condition is reached.

    This will be done ``cycles`` times (supporting half cycles for ending after
    the first condition).
  """

  def __init__(self, time, cmd, condition1, condition2, value1, value2,
               cycles=1, verbose=False):
    """Sets the args and initializes parent class.

    Args:
      time:
      cmd:
      condition1 (:obj:`str`): Representing the condition to switch to
        ``value2``. See :ref:`generator path` for more info.
      condition2 (:obj:`str`): Representing the condition to switch to
        ``value1``. See :ref:`generator path` for more info.
      value1: First value to send.
      value2: Second value to send.
      cycles: Number of time we should be doing this.

        Note:
          ``cycles = 0`` will make it loop forever.

      verbose:

    Note:
      ::

        [{'type': 'cyclic', 'value1': 1, 'condition1': 'AIN0>2',
        'value2': 0, 'condition2': 'AIN1<1', 'cycles': 5}]

      is equivalent to
      ::

        [{'type': 'constant', 'value': 1,'condition': 'AIN0>2'},
        {'type': 'constant', 'value': 0, 'condition': 'AIN1<1'}] * 5
    """

    Path.__init__(self, time, cmd)
    self.value = (value1, value2)
    self.condition1 = self.parse_condition(condition1)
    self.condition2 = self.parse_condition(condition2)
    self.cycles = int(2 * cycles)  # Logic in this class will be in half-cycle
    self.cycle = 0
    self.verbose = verbose

  def get_cmd(self, data):
    if 0 < self.cycles <= self.cycle:
      raise StopIteration
    if not self.cycle % 2 and self.condition1(data) or self.cycle % 2 \
          and self.condition2(data):
      self.cycle += 1
      if self.verbose:
        print("cyclic path {}/{}".format(self.cycle, self.cycles))
      self.t0 = time()
    return self.value[self.cycle % 2]
