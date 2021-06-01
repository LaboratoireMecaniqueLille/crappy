# coding: utf-8

from time import time

from .block import Block


class PID(Block):
  """
  A PID corrector.

  Note:
    A PID will continuously adjust the output based on the target
    and the actual value, to try to actually reach the target.

  Args:
    - kp (float): P gain.
    - ki (float, default: 0): I gain.
    - kd (float, default: 0): D gain.

  Kwargs:
    - out_max (float or None, default: None): If not None, it will always keep
      the output below this value.
    - out_min (float or None, default: None): If not None, it will always keep
      the output above this value.
    - target_label (str, default: "cmd"): The label of the setpoint.
    - input_label (str, default: "V"): The reading of the actual value to be
      compared with the setpoint.
    - time_label (str, default: "t(s)"): The label of the time.
    - labels (list, default: ['t(s)', 'pid']): The labels of the output of the
      block.

      Note:
        It must have TWO strings: the time and the command.

    - reverse (bool, default: False): To reverse the retro action.
    - i_limit (float / tuple, default: 1): To avoid over integration.

      Note:
        If it is a tuple, it is the boundaries for the I term.

        If it is a float, the boundaries will be i_limit * out_min,
        i_limit * out_max (typically between 0 and 1)

    - send_terms (bool, default: False): To get the weight of each term in the
      output value.

      Note:
        It will add ['p_term', 'i_term', 'd_term'] to the labels.

        This is particularly useful to tweak the gains.

  """

  def __init__(self, kp, ki=0, kd=0, **kwargs):
    Block.__init__(self)
    self.niceness = -10
    for arg, default in [('freq', 500),
                    ('out_max', float('inf')),
                    ('out_min', -float('inf')),
                    ('target_label', 'cmd'),
                    ('input_label', 'V'),
                    ('time_label', 't(s)'),
                    ('labels', ['t(s)', 'pid']),
                    ('reverse', False),
                    ('i_limit', 1),
                    ('send_terms', False)  # For debug, mostly
    ]:
      setattr(self, arg, kwargs.pop(arg, default))
    assert not kwargs, "PID got incorrect kwarg(s): " + str(kwargs)
    self.set_k(kp, ki, kd)
    self.i_term = 0
    self.last_val = 0
    if self.send_terms:
      self.labels.extend(['p_term', 'i_term', 'd_term'])
    if not isinstance(self.i_limit, tuple):
      i_min = self.i_limit * self.out_min if self.out_min is not None else None
      i_max = self.i_limit * self.out_max if self.out_max is not None else None
      self.i_limit = (i_min, i_max)
    assert len(self.i_limit) == 2, "Invalid i_limit arg!"

  def begin(self):
    self.last_t = self.t0
    data = [inp.recv_last(True) for inp in self.inputs]
    for i, r in enumerate(data):
      if self.target_label in r:
        self.target_link_id = i
      if self.input_label in r and self.time_label in r:
        self.feedback_link_id = i
    assert hasattr(self, "target_link_id"), "[PID] Error: no link containing"\
        " target label {}".format(self.target_label)
    assert hasattr(self, "feedback_link_id"), \
      "[PID] Error: no link containing input label {} " \
      "and time label {}".format(self.input_label, self.time_label)
    assert set(range(len(self.inputs))) == {(self.target_link_id,
                                             self.feedback_link_id)}, \
      "[PID] Error: useless link(s)! Make sure PID block does not " \
      "have extra inputs"
    self.last_target = data[self.target_link_id][self.target_label]
    self.last_t = data[self.feedback_link_id][self.time_label]
    # For the classical D approach:
    # self.last_val = self.last_target -\
    #    data[self.feedback_link_id][self.input_label]
    # When ignore setpoint mode
    self.last_val = data[self.feedback_link_id][self.input_label]

    if self.send_terms:
      self.send([self.last_t, 0, 0, 0, 0])
    else:
      self.send([self.last_t, 0])

  def clamp(self, v, limits=None):
    if limits is None:
      mini, maxi = self.out_min, self.out_max
    else:
      mini, maxi = limits
    return max(v if maxi is None else min(v, maxi), mini)

  def set_k(self, kp, ki=0, kd=0):
    s = -1 if self.reverse else 1
    self.kp = s * kp
    self.ki = s * kp * ki
    self.kd = s * kp * kd

  def loop(self):
    data = self.inputs[self.feedback_link_id].recv_last(True)
    t = data[self.time_label]
    dt = t - self.last_t
    if dt <= 0:
      return
    feedback = data[self.input_label]
    if self.feedback_link_id == self.target_link_id:
      target = data[self.target_label]
    else:
      data = self.inputs[self.target_link_id].recv_last()
      if data is None:
        target = self.last_target
      else:
        target = data[self.target_label]
        self.last_target = target
    diff = target-feedback

    p_term = self.kp*diff
    self.last_t = t

    # Classical approach
    # d_term = self.kd * (diff - self.last_val)
    # self.last_val = diff
    # Alternative: ignore setpoint to avoid derivative kick
    d_term = -self.kd * (feedback - self.last_val) / dt
    self.last_val = feedback

    self.i_term += self.ki * diff * dt
    out = p_term + self.i_term + d_term
    if not self.out_min < out < self.out_max:
      out = self.clamp(out)
    self.i_term = self.clamp(self.i_term, self.i_limit)
    if self.send_terms:
      self.send([time() - self.t0, out, p_term, self.i_term, d_term])
    else:
      self.send([time() - self.t0, out])
