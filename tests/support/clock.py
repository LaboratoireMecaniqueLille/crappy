"""A local clock for deterministic calculations and simulated I/O delays."""


class Clock:
  def __init__(self, now=0.0):
    self.now = now
    self.sleeps = []

  def __call__(self):
    return self.now

  def advance(self, seconds):
    self.now += seconds

  def sleep(self, seconds):
    if seconds < 0:
      raise AssertionError('a test requested a negative wait')
    self.sleeps.append(seconds)
    self.advance(seconds)
