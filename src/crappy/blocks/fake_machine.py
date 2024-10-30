# coding: utf-8

from time import time
import numpy as np
from typing import Optional, Literal
from collections.abc import Callable
import logging

from .meta_block import Block


def plastic(v: float, yield_strain: float = .005, rate: float = .02) -> float:
  """A basic plastic law given as an example."""

  if v > yield_strain:
    return rate * ((((v - yield_strain) / rate) ** 2 + 1) ** .5 - 1)
  return 0


class FakeMachine(Block):
  """This Block emulates the behavior of a tensile test machine.
  
  It can emulate tensile tests, **not compression tests**. By default, it 
  assumes an elasto-plastic behavior of the tested sample. The main mechanical
  parameters of the material are tunable.
  
  This Block is meant to be driven like a :class:`~crappy.blocks.Machine` 
  Block. It receives speed or position commands from upstream Blocks, and 
  modifies the behavior of the emulated machine accordingly. Its outputs are
  however different from the Machine Blocks, as it outputs the current force, 
  position, and strain of the emulated tensile test machine. The labels
  carrying this data are : ``t(s), F(N), x(mm), Exx(%), Eyy(%)``.
  
  This Block was originally designed for proposing examples that do not require
  any hardware to run, but still display the possibilities of Crappy. It can
  also be used to test a script without actually interacting with hardware.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Fake_machine* to *FakeMachine*
  """

  def __init__(self,
               rigidity: float = 8.4E6,
               l0: float = 200,
               max_strain: float = 1.51,
               sigma: Optional[dict[str, float]] = None,
               nu: float = 0.3,
               plastic_law: Callable[[float], float] = plastic,
               max_speed: float = 5,
               mode: Literal['speed', 'position'] = 'speed',
               cmd_label: str = 'cmd',
               freq: Optional[float] = 100,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      rigidity: The rigidity of the material, in N, so that 
        :math:`force = rigidity * strain`.

        .. versionchanged:: 2.0.0 renamed from *k* to *rigidity*
      l0: The initial length of the fake sample to test, in mm.
      max_strain: The maximum strain the material can withstand before
        breaking.

        .. versionchanged:: 1.5.10 renamed from *maxstrain* to *max_strain*
      mode: Whether the command sent to the fake machine is a speed or a
        position command. Can be ``'speed'`` or ``'position'``.
      plastic_law: A callable taking the maximum reached strain and returning
        the proportion of the current strain caused by plastic deformation.
      sigma: A :obj:`dict` containing for each label the standard deviation for
        adding noise to the signal. Can be given for part or all of the labels.
        The deviation should be given not normalized, in the same unit as the
        label to which it applies.
      nu: Poisson's ratio of the material.
      cmd_label: The label carrying the command of the fake machine.
      freq: The target looping frequency for the Block. If :obj:`None`, loops 
        as fast as possible.

        .. versionadded:: 1.5.10
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

    super().__init__()
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    # Setting the mechanical parameters of the material
    self._rigidity = rigidity
    self._l0 = l0
    self._max_strain = max_strain / 100
    self._nu = nu
    self._max_speed = max_speed
    self._sigma = {'F(N)': 50, 'x(mm)': 2e-3,
                   'Exx(%)': 1e-3, 'Eyy(%)': 1e-3} if sigma is None else sigma
    self._plastic_law = plastic_law

    self._mode = mode
    self._cmd_label = cmd_label

    # Creating the mechanical variables
    self._current_pos = 0
    self._prev_t = None
    self._prev_broke_t = time()
    self._plastic_elongation = 0
    self._max_recorded_strain = 0

  def begin(self) -> None:
    """Sends a first value that should be 0."""

    self._prev_t = self.t0
    self._send_values()

  def loop(self) -> None:
    """Receives the latest command value, calculates the new speed and position
    from it, checks whether the sample broke and what the plastic elongation
    is, and finally returns the data."""

    # Getting the latest command
    if self._cmd_label not in (data := self.recv_last_data(fill_missing=True)):
      return
    else:
      cmd = data[self._cmd_label]
    t = time()
    delta_t = t - self._prev_t
    self._prev_t = t

    # Calculating the speed based on the command and the mode
    if self._mode == 'speed':
      speed = float(np.sign(cmd)) * float(np.min((self._max_speed,
                                                  np.abs(cmd))))
    elif self._mode == 'position':
      speed = float(np.sign(cmd - self._current_pos)) * float(np.min(
          (self._max_speed, np.abs(cmd - self._current_pos) / delta_t)))
    else:
      raise ValueError(f'Invalid mode : {self._mode} !')

    # Updating the current position
    self._current_pos += speed * delta_t

    # If the max strain is reached, consider that the sample broke
    if self._current_pos / self._l0 > self._max_strain:
      if time() - self._prev_broke_t > 1:
        self._prev_broke_t = time()
        self.log(logging.WARNING, "Sample broke !")
      self._rigidity = 0

    # Compute the plastic elongation separately
    if self._current_pos / self._l0 > self._max_recorded_strain:
      self._max_recorded_strain = self._current_pos / self._l0
      self._plastic_elongation = self._plastic_law(
        self._max_recorded_strain) * self._l0

    # Finally, sending the values
    self._send_values()

  def _add_noise(self, to_send: dict[str, float]) -> dict[str, float]:
    """Adds noise to the data to be sent, according to the sigma values
    provided by the user. Then returns the noised data."""

    for label, value in to_send.items():
      if label in self._sigma:
        to_send[label] = np.random.normal(value, self._sigma[label])

    return to_send

  def _send_values(self) -> None:
    """Gathers all the information to be sent, adds noise and send it."""

    to_send = {'t(s)': time() - self.t0,
               'F(N)': (self._current_pos -
                        self._plastic_elongation) / self._l0 * self._rigidity,
               'x(mm)': self._current_pos,
               'Exx(%)': self._current_pos * 100 / self._l0,
               'Eyy(%)': -self._nu * self._current_pos * 100 / self._l0}

    self.send(self._add_noise(to_send))
