# coding: utf-8

from time import time
from typing import List, Optional, Tuple

from .block import Block


class PID(Block):
  """A basic implementation of a PID corrector.

  A PID will continuously adjust its output based on the target value and the
  actual measured value, to try to actually reach the target.
  """

  def __init__(self,
               kp: float,
               ki: float = 0,
               kd: float = 0,
               out_max: float = float('inf'),
               out_min: float = -float('inf'),
               target_label: str = 'cmd',
               input_label: str = 'V',
               time_label: str = 't(s)',
               labels: Optional[List[str]] = None,
               reverse: bool = False,
               i_limit: Tuple[Optional[float], Optional[float]] = (None, None),
               send_terms: bool = False,
               freq: float = 500,
               verbose: bool = False) -> None:
    """Sets the args and initializes the parent class.

    Args:
      kp: The `P` gain.
      ki: The `I` gain.
      kd: The `D` gain.
      out_max: Ensures the output is always inferior to this value.
      out_min: Ensures the output is always superior to this value.
      target_label: The label carrying the setpoint value.
      input_label: The label carrying the reading of the actual value, to be
        compared with the setpoint.
      time_label: The label carrying the time information in the incoming
        links.
      labels: The two labels that will be sent to downstream blocks. The first
        one is the time label, the second one is the output of the PID. If this
        argument is not given, they default to ``'t(s)'`` and ``'pid'``.
      reverse: If :obj:`True`, reverses the action of the PID.
      i_limit: A :obj:`tuple` containing respectively the lower and upper
        boundaries for the `I` term.
      send_terms: If :obj:`True`, returns the weight of each term in the output
        value. It adds ``'p_term', 'i_term', 'd_term'`` to the output labels.
        This is particularly useful to tweak the gains.
      freq: The block will try to loop at this frequency.
      verbose: If :obj:`True`, prints the looping frequency of the block.
    """

    # Attributes of the parent class
    super().__init__()
    self.niceness = -10
    self.freq = freq
    self.verbose = verbose
    self.labels = ['t(s)', 'pid'] if labels is None else labels
    if send_terms:
      self.labels.extend(['p_term', 'i_term', 'd_term'])

    # Setting the gains
    sign = -1 if reverse else 1
    self._kp = sign * kp
    self._ki = sign * kp * ki
    self._kd = sign * kp * kd

    # Setting the limits
    self._out_max = out_max
    self._out_min = out_min
    self._i_min, self._i_max = i_limit

    # Setting the labels
    self._target_label = target_label
    self._input_label = input_label
    self._time_label = time_label

    self._send_terms = send_terms

    # Setting the variables
    self._target = None
    self._last_input = None
    self._last_t = None
    self._i_term = 0

  def begin(self) -> None:
    """Receives the first target and input values and makes sure the given
    labels are correct."""

    # Waiting for the first data to arrive
    data = [link.recv(blocking=True) for link in self.inputs]

    # Getting the first target value
    for dic in data:
      if self._target_label in dic:
        self._target = dic[self._target_label]
        break

    # Getting the first input value
    for dic in data:
      if self._input_label in dic and self._time_label in dic:
        self._last_input = dic[self._input_label]
        self._last_t = dic[self._time_label]
        break

    # Making sure the given labels are correct
    if self._target is None:
      raise IOError(f'No link containing the target label '
                    f'{self._target_label} !')
    if self._last_input is None:
      raise IOError(f'No link containing the input label {self._input_label} '
                    f'and the time label {self._time_label} !')

  def loop(self) -> None:
    """Receives the latest target and input values, calculates the P, I and D
    terms and sends the output to the downstream blocks."""

    # Looping in a non-blocking way
    data = [link.recv_last(blocking=False) for link in self.inputs]

    input_ = None
    t = None

    # Updating the target value
    for dic in data:
      if dic is not None and self._target_label in dic:
        self._target = dic[self._target_label]
        break

    # Getting the latest input value
    for dic in data:
      if dic is not None and self._input_label in dic \
            and self._time_label in dic:
        input_ = dic[self._input_label]
        t = dic[self._time_label]
        break

    # If there's no new input, do nothing
    if input_ is None or t is None:
      return

    delta_t = t - self._last_t
    diff = self._target - input_

    # Calculating the three PID terms
    p_term = self._kp * diff
    self._i_term += self._ki * diff * delta_t
    d_term = - self._kd * (input_ - self._last_input) / delta_t

    self._last_t = t
    self._last_input = input_

    # Clamping the i term if required
    if self._i_min is not None:
      self._i_term = max(self._i_min, self._i_term)
    if self._i_max is not None:
      self._i_term = min(self._i_max, self._i_term)

    # Clamping the output if required
    out = p_term + self._i_term + d_term
    out = min(self._out_max, max(self._out_min, out))

    # Sending the values to the downstream blocks
    if self._send_terms:
      self.send([time() - self.t0, out, p_term, self._i_term, d_term])
    else:
      self.send([time() - self.t0, out])
