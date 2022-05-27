# coding: utf-8

from time import time
import numpy as np
from typing import Callable, Literal, Dict, Optional

from .block import Block


def plastic(v: float, yield_strain: float = .005, rate: float = .02) -> float:
  """A basic plastic law given as an example."""

  if v > yield_strain:
    return rate * ((((v - yield_strain) / rate) ** 2 + 1) ** .5 - 1)
  return 0


class Fake_machine(Block):
  """This block simulates the behavior of a tensile test machine.

  It should be used to simulate tensile teste, not compression tests. By
  default, it assumes a plastic behavior of the material. The main mechanical
  parameters of the material are tunable.

  This block is meant to be driven like the :ref:`Machine` block. However, its
  outputs are different and are : ``t(s), F(N), x(mm), Exx(%), Eyy(%)``.
  """

  def __init__(self,
               k: float = 8.4E6,
               l0: float = 200,
               max_strain: float = 1.51,
               sigma: Optional[Dict[str, float]] = None,
               nu: float = 0.3,
               plastic_law: Callable[[float], float] = plastic,
               max_speed: float = 5,
               mode: Literal['speed', 'position'] = 'speed',
               cmd_label: str = 'cmd',
               freq: float = 100,
               verbose: bool = False) -> None:
    """Sets the args and initializes the parent class.

    Args:
      k: The rigidity of the material, in N, so that ``force = k x strain``.
      l0: The initial length of the fake sample to test, in mm.
      max_strain: The maximum strain the material can withstand before
        breaking.
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
      freq: The block will try to loop at this frequency.
      verbose: If :obj:`True`, prints the looping frequency of the block.
    """

    super().__init__()
    self.freq = freq
    self.verbose = verbose

    # Setting the mechanical parameters of the material
    self._k = k
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
    self._last_t = None
    self._plastic_elongation = 0
    self._max_recorded_strain = 0

  def prepare(self) -> None:
    """Sends a value so that the camera gets an image during prepare for the
    fake test examples. To be removed at some point."""

    # Todo: To be removed when compute_strain_img is replaced with a getter
    self.t0 = time()
    self._send_values()

  def begin(self) -> None:
    """Sends a first value that should be 0."""

    self._last_t = self.t0
    self._send_values()

  def loop(self) -> None:
    """Receives the latest command value, calculates the new speed and position
    from it, checks whether the sample broke and what the plastic elongation
    is, and finally returns the data"""

    # Getting the latest command
    cmd = self.get_last()[self._cmd_label]
    t = time()
    delta_t = t - self._last_t
    self._last_t = t

    # Calculating the speed based on the command and the mode
    if self._mode == 'speed':
      speed = np.sign(cmd) * np.min((self._max_speed, np.abs(cmd)))
    elif self._mode == 'position':
      speed = np.sign(cmd - self._current_pos) * np.min(
          (self._max_speed, np.abs(cmd - self._current_pos) / delta_t))
    else:
      raise ValueError(f'Invalid mode : {self._mode} !')

    # Updating the current position
    self._current_pos += speed * delta_t

    # If the max strain is reached, consider that the sample broke
    if self._current_pos / self._l0 > self._max_strain:
      self._k = 0

    # Compute the plastic elongation separately
    if self._current_pos / self._l0 > self._max_recorded_strain:
      self._max_recorded_strain = self._current_pos / self._l0
      self._plastic_elongation = self._plastic_law(
        self._max_recorded_strain) * self._l0

    # Finally, sending the values
    self._send_values()

  def _add_noise(self, to_send: Dict[str, float]) -> Dict[str, float]:
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
                        self._plastic_elongation) / self._l0 * self._k,
               'x(mm)': self._current_pos,
               'Exx(%)': self._current_pos * 100 / self._l0,
               'Eyy(%)': -self._nu * self._current_pos * 100 / self._l0}

    self.send(self._add_noise(to_send))
