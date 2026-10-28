# coding: utf-8

import json
from pathlib import Path
from time import monotonic, time
from multiprocessing import Value
from crappy.blocks.generator_path.meta_path import Path as GeneratorPath
from tests.support.processes import wait_until_stable

import crappy
from crappy.blocks import Block


class CoordinateSource(Block):
  """Outputs deterministic marker coordinates for auto-drive testing."""

  def __init__(self) -> None:
    """Sets the source labels and loop frequency."""

    super().__init__()
    self.labels = ['t(s)', 'Coord(px)']
    self.freq = 30
    self.pausable = False

  def loop(self) -> None:
    """Sends two marker coordinates moving together horizontally."""

    elapsed = time() - self.t0
    offset = 5 * elapsed
    coordinates = [(20., 25. + offset), (40., 45. + offset)]
    self.send([elapsed, coordinates])


class ScriptedControlSource(Block):
  """Drive pause/resume/stop after observed progress, with a hard deadline."""

  def __init__(self, counter, path):
    super().__init__()
    self.labels = ['t(s)', 'pause', 'stop']
    self.freq, self.pausable = 200, False
    self._counter, self._path = counter, path
    self._stage = 0
    self._observations = {}

  def begin(self):
    self._deadline = monotonic() + 10

  def loop(self):
    if monotonic() > self._deadline:
      raise RuntimeError('pause/resume control did not make progress')
    if self._stage == 0:
      if self._counter.value < 2:
        return
      self._observations['before_pause'] = self._counter.value
      self._stage = 1
    if self._stage == 1:
      self.send([time() - self.t0, 1, 0])
      if not self._pause_event.is_set():
        return
      self._observations['pause_observed'] = True
      settled = wait_until_stable(lambda: self._counter.value, timeout=3)
      self._observations['paused_start'] = settled
      self._stop_event.wait(0.15)
      self._observations['paused_end'] = self._counter.value
      self._stage = 2
    if self._stage == 2:
      self.send([time() - self.t0, 0, 0])
      if self._pause_event.is_set() or self._counter.value < self._observations['paused_end'] + 2:
        return
      self._observations['resumed'] = self._counter.value
      self._stage = 3
    self.send([time() - self.t0, 0, 1])

  def finish(self):
    self._path.write_text(json.dumps(self._observations), encoding='utf-8')


class LifecycleProbe(Block):
  """Record user loop progress and finalization to a test-owned artifact."""

  def __init__(self, path, counter):
    super().__init__()
    self.freq = 50
    self._path, self._counter = path, counter
    self._loop_times = []

  def loop(self):
    self._loop_times.append(time() - self.t0)
    with self._counter.get_lock():
      self._counter.value += 1

  def finish(self):
    self._path.write_text(json.dumps({'loop_times': self._loop_times,
                                     'finish_called': True}), encoding='utf-8')


def build_auto_drive_recorder(output_dir: Path) -> tuple[Block, ...]:
  """Builds a coordinate source -> AutoDrive -> Recorder script."""

  source = CoordinateSource()

  auto_drive = crappy.blocks.AutoDriveVideoExtenso(
    {'type': 'FakeDCMotor', 'inertia': 1, 'kv': 100},
    gain=2,
    direction='X+',
    pixel_range=100,
    max_speed=100,
    freq=30)

  recorder = crappy.blocks.Recorder(
    output_dir / 'auto_drive.csv',
    labels=('t(s)', 'diff(pix)'),
    delay=0.1,
    freq=20)

  stop = crappy.blocks.StopBlock('t(s) > 0.8', freq=50)

  crappy.link(source, auto_drive)
  crappy.link(auto_drive, recorder)

  return source, auto_drive, recorder, stop


def build_pause_stop_probe(output_dir: Path) -> tuple[Block, ...]:
  """Builds a scripted source -> Pause/StopBlock script with a probe."""

  counter = Value('i', 0)
  source = ScriptedControlSource(counter, output_dir / 'pause_controls.json')
  pause = crappy.blocks.Pause('pause > 0.5', freq=20)
  stop = crappy.blocks.StopBlock('stop > 0.5', freq=50)
  probe = LifecycleProbe(output_dir / 'pause_probe.json', counter)

  crappy.link(source, pause)
  crappy.link(source, stop)

  return source, pause, stop, probe


def build_generator_sink(_: Path) -> tuple[Block, ...]:
  """Builds a finite Generator -> Sink lifecycle script."""

  generator = crappy.blocks.Generator(
    ({'type': 'Constant', 'value': 1, 'condition': 'delay=0.5'},),
    spam=True,
    end_delay=0.2,
    freq=20)
  sink = crappy.blocks.Sink(freq=20)

  crappy.link(generator, sink)

  return generator, sink


def build_generator_link_reader(_: Path) -> tuple[Block, ...]:
  """Builds a finite Generator -> LinkReader lifecycle script."""

  generator = crappy.blocks.Generator(
    ({'type': 'Ramp',
      'speed': 1,
      'condition': 'delay=0.5',
      'init_value': 0},),
    cmd_label='value',
    spam=True,
    end_delay=0.2,
    freq=10)
  reader = crappy.blocks.LinkReader(name='Integration reader', freq=20)

  crappy.link(generator, reader)

  return generator, reader


class RepeatingIntegrationPath(GeneratorPath):
  """A finite Path whose completion depends on samples, not scheduling speed."""

  def __init__(self):
    super().__init__()
    self.remaining = 2

  def get_cmd(self, data):
    if not self.remaining:
      raise StopIteration
    self.remaining -= 1
    return 3


class PhaseRecorder(Block):
  def __init__(self, path):
    super().__init__()
    self.freq = 100
    self.path, self.rows = path, []

  def loop(self):
    data = self.recv_all_data()
    if data:
      self.rows.extend(zip(data['t(s)'], data['cmd'], data['phase']))

  def finish(self):
    self.loop()
    self.path.write_text(json.dumps(self.rows), encoding='utf-8')


def build_repeating_generator(output_dir):
  generator = crappy.blocks.Generator(
    ({'type': 'RepeatingIntegrationPath'},), repeat=True, end_delay=None,
    cmd_label='cmd', path_index_label='phase', freq=100)
  recorder = PhaseRecorder(output_dir / 'repeat.json')
  stop = crappy.blocks.StopBlock('phase > 1.5', freq=200)
  crappy.link(generator, recorder)
  crappy.link(generator, stop)
  return generator, recorder, stop
