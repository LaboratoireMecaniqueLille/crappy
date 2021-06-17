from time import time
import numpy as np

from .block import Block


def plastic(v, yield_strain=.005, rate=.02):
  if v > yield_strain:
    return ((v - yield_strain) ** 2 + rate ** 2) ** .5 - rate
  return 0


class Fake_machine(Block):
  """Block to simulate the behavior of a tensile testing machine.

  By default, it is supposed to represent a material with a plastic behavior.
  """

  def __init__(self,
               k=210000*20*2,  # Global rigidity in N (Force = k*strain)
               l0=200,  # Initial length of the sample
               maxstrain=1.51,  # The material breaks above this strain (%)
               mode='speed',  # Does the machine take the input in speed
               # or position
               max_speed=5,  # mm/s
               plastic_law=plastic,  # Returns the plastic strain given
               # a strain
               # To add normal noise over the data and make things a bit
               # more realistic!
               sigma=None,
               nu=.3,
               cmd_label='cmd'):
    Block.__init__(self)
    self.freq = 100
    self.k = k
    self.l0 = l0
    self.maxstrain = maxstrain/100
    self.mode = mode
    self.max_speed = max_speed
    self.nu = nu
    self.cmd_label = cmd_label
    self.pos = 0
    self.last_t = None
    self.plastic_elongation = 0
    self.plastic_law = plastic_law
    self.sigma = {'F(N)': 50, 'x(mm)': 2e-3, 'Exx(%)': 1e-3, 'Eyy(%)': 1e-3} \
      if sigma is None else sigma
    self.max_seen_strain = 0

  def noise(self, d):
    for k in d:
      if k in self.sigma:
        d[k] = np.random.normal(d[k], self.sigma[k])
    return d

  def send_all(self):
    tosend = {
        't(s)': time() - self.t0,
        'F(N)': (self.pos - self.plastic_elongation) / self.l0 * self.k,
        'x(mm)': self.pos,
        'Exx(%)': self.pos * 100 / self.l0
      }
    tosend['Eyy(%)'] = -self.nu*tosend['Exx(%)']
    self.send(self.noise(tosend))

  def prepare(self):
    self.t0 = time()
    self.send_all()

  def begin(self):
    self.last_t = self.t0
    self.send_all()

  def loop(self):
    cmd = self.get_last()[self.cmd_label]
    t = time()
    dt = t - self.last_t
    if dt < 0:
      return
    if self.mode == 'speed':
      speed = np.sign(cmd) * np.min((self.max_speed, np.abs(cmd)))
    elif self.mode == 'position':
      speed = np.sign(cmd - self.pos) * np.min(
          (self.max_speed, np.abs(cmd - self.pos) / dt))
    else:
      raise AttributeError("Unknown mode:" + str(self.mode))
    self.pos += dt * speed
    if self.pos / self.l0 > self.maxstrain:
      self.k = 0
    if self.pos / self.l0 > self.max_seen_strain:
      self.max_seen_strain = self.pos / self.l0
      self.plastic_elongation = self.plastic_law(self.max_seen_strain) * \
                                self.l0
    self.send_all()
    self.last_t = t
