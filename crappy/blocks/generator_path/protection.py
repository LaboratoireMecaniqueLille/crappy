# coding: utf-8

from .path import Path


class Protection(Path):
  """Useful to protect samples from being pulled apart when setting up a test.
  """

  def __init__(self, time, cmd, condition1, condition2, value1, value2,
               value0=0, verbose=False):
    """Sets the args and initializes parent class.

    Args:
      time:
      cmd:
      condition1 (:obj:`str`): Representing the first condition. See
        :ref:`generator path` for more info.
      condition2 (:obj:`str`): Representing the second condition. See
        :ref:`generator path` for more info.
      value1: Value to send when ``condition1`` is met.
      value2: Value to send when ``condition2`` is met.
      value0: Value to send when no condition is reached.
      verbose:

    Note:
      By default will send ``value0``.

      While ``condition1`` is met, will return ``value1``.

      While ``condition2`` is met, will return ``value2``.

      If ``condition1`` and ``condition2`` are met simultaneously, the first
      one met will prevail. If met at the same time, ``condition1`` will
      prevail.
    """

    Path.__init__(self, time, cmd)
    self.value = (value0, value1, value2)
    self.condition1 = self.parse_condition(condition1)
    self.condition2 = self.parse_condition(condition2)
    s = '<' if '<' in condition1 else '>'
    self.lbl1 = condition1.split(s)[0]
    s = '<' if '<' in condition2 else '>'
    self.lbl2 = condition2.split(s)[0]
    self.verbose = verbose
    self.status = 0

  def get_cmd(self, data):
    if self.status == 0:
      if self.condition1(data):
        self.status = 1
      elif self.condition2(data):
        self.status = 2
      return self.value[self.status]
    if self.status == 1 and data[self.lbl1] and not self.condition1(data):
      self.status = 0
    elif self.status == 2 and data[self.lbl2] and not self.condition2(data):
      self.status = 0
    return self.value[self.status]
