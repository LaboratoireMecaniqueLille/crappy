# coding: utf-8

from time import time
from typing import Optional
import logging

from .meta_block import Block


class PID(Block):
  """This Block implements a basic PID corrector.
  
  It takes a target value and a measured value as an input, and outputs a
  command value for reaching the target on the controlled system. The command
  is updated at each loop. The given `P`, `I` and `D` gains can be adjusted at
  runtime, by sending the new desired values. It is also possible to set 
  boundaries to the output command, or to the integral corrector only.
  
  This Block is generally used in combination with a 
  :class:`~crappy.blocks.Generator` Block for generating the target value, and
  a :class:`~crappy.blocks.Machine` or :class:`~crappy.blocks.IOBlock` Block
  driving a physical system. This latter Block takes the command as input, and
  returns to the PID the measured value to be compared to the target.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               kp: float,
               ki: float = 0,
               kd: float = 0,
               out_max: float = float('inf'),
               out_min: float = -float('inf'),
               setpoint_label: str = 'cmd',
               input_label: str = 'V',
               time_label: str = 't(s)',
               kp_label: str = 'kp',
               ki_label: str = 'ki',
               kd_label: str = 'kd',
               labels: Optional[tuple[str, str]] = None,
               reverse: bool = False,
               i_limit: tuple[Optional[float], Optional[float]] = (None, None),
               send_terms: bool = False,
               freq: Optional[float] = 500,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      kp: The initial `P` gain. It can be tuned while running by sending the
        new value over the given ``kp_label``. No matter if a positive or a
        negative value is given, the definitive sign will be set by the
        ``reverse`` argument.
      ki: The initial `I` gain. It can be tuned while running by sending the
        new value over the given ``ki_label``. No matter if a positive or a
        negative value is given, the definitive sign will be set by the
        ``reverse`` argument.
      kd: The initial `D` gain. It can be tuned while running by sending the
        new value over the given ``kd_label``. No matter if a positive or a
        negative value is given, the definitive sign will be set by the
        ``reverse`` argument.
      out_max: Ensures the output is always inferior to this value.
      out_min: Ensures the output is always superior to this value.
      setpoint_label: The label carrying the setpoint value.

        .. versionchanged:: 2.0.0
           renamed from *target_label* to *setpoint_label*
      input_label: The label carrying the reading of the actual value, to be
        compared with the setpoint.
      time_label: The label carrying the time information in the incoming
        Links.
      kp_label: The label to use for changing the `P` gain on the fly. If a
        value is received over this label, it will overwrite the one given in
        the ``kp`` argument.

        .. versionadded:: 2.0.0
      ki_label: The label to use for changing the `I` gain on the fly. If a
        value is received over this label, it will overwrite the one given in
        the ``ki`` argument.

        .. versionadded:: 2.0.0
      kd_label: The label to use for changing the `D` gain on the fly. If a
        value is received over this label, it will overwrite the one given in
        the ``kd`` argument.

        .. versionadded:: 2.0.0
      labels: The two labels that will be sent to downstream Blocks. The first
        one is the time label, the second one is the output of the PID. If this
        argument is not given, they default to ``'t(s)'`` and ``'pid'``.
      reverse: If :obj:`True`, reverses the action of the PID.
      i_limit: A :obj:`tuple` containing respectively the lower and upper
        boundaries for the `I` term.
      send_terms: If :obj:`True`, returns the weight of each term in the output
        value. It adds ``'p_term', 'i_term', 'd_term'`` to the output labels.
        This is particularly useful to tweak the gains.
      freq: The target looping frequency for the Block. If :obj:`None`, loops 
        as fast as possible.
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
        
        .. versionadded:: 1.5.10
        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
        
        .. versionadded:: 2.0.0
    """

    # Attributes of the parent class
    super().__init__()
    self.niceness = -10
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug
    self.labels = ['t(s)', 'pid'] if labels is None else list(labels)
    if send_terms:
      self.labels.extend(['p_term', 'i_term', 'd_term'])

    # Setting the gains
    sign = -1 if reverse else 1
    self._kp = sign * abs(kp)
    self._ki = sign * abs(ki)
    self._kd = sign * abs(kd)

    # Setting the limits
    self._out_max = out_max
    self._out_min = out_min
    self._i_min, self._i_max = i_limit

    # Setting the labels
    self._target_label = setpoint_label
    self._input_label = input_label
    self._time_label = time_label
    self._kp_label = kp_label
    self._ki_label = ki_label
    self._kd_label = kd_label

    self._send_terms = send_terms
    self._reverse = reverse

    # Setting the variables
    self._setpoint: Optional[float] = None
    self._last_input: Optional[float] = None
    self._prev_t: float = 0.
    self._i_term: float = 0.

  def loop(self) -> None:
    """Receives the latest target and input values, calculates the P, I and D
    terms and sends the output to the downstream Blocks."""

    # Looping in a non-blocking way
    data = self.recv_last_data(fill_missing=False)

    # Updating the gains if provided
    if self._kp_label in data:
      kp = data[self._kp_label]
      self._kp = -abs(kp) if self._reverse else kp
    if self._ki_label in data:
      ki = data[self._ki_label]
      self._ki = -abs(ki) if self._reverse else ki
    if self._kd_label in data:
      kd = data[self._kd_label]
      self._kd = -abs(kd) if self._reverse else kd

    # Updating the target value if provided
    if self._target_label in data:
      self._setpoint = data[self._target_label]
      self.log(logging.DEBUG, f"Updated target value to {self._setpoint}")

    # Checking if a new input was received
    if self._time_label in data and self._input_label in data:
      input_ = data[self._input_label]
      t = data[self._time_label]
      self.log(logging.DEBUG, f"Updated input value to {input_} at time {t}")

      # For the first loops, setting the target to the first input by default
      if self._setpoint is None:
        self._setpoint = input_

      # For the first loops, initializing the input history
      if self._last_input is None:
        self._last_input = input_

    # No new input was received
    else:
      return

    delta_t = t - self._prev_t
    error = self._setpoint - input_
    d_input = input_ - self._last_input

    # Calculating the three PID terms
    p_term = self._kp * error
    self._i_term += self._ki * error * delta_t
    d_term = - self._kd * d_input / delta_t if delta_t > 0 else 0

    self._prev_t = t
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
