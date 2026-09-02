# coding: utf-8

import json
from pathlib import Path
from time import time

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
  """Drives deterministic pause, resume, and stop transitions."""

  def __init__(self) -> None:
    """Sets the source labels and keeps it active during pauses."""

    super().__init__()
    self.labels = ['t(s)', 'pause', 'stop']
    self.freq = 200
    self.pausable = False

  def loop(self) -> None:
    """Sends a pause pulse followed later by a stop pulse."""

    elapsed = time() - self.t0
    pause = int(0.2 <= elapsed < 0.5)
    stop = int(elapsed >= 0.8)
    self.send([elapsed, pause, stop])


class LifecycleProbe(Block):
  """Records its active loop timestamps to a JSON artifact."""

  def __init__(self, path: Path) -> None:
    """Stores the artifact path and initializes the timestamp buffer."""

    super().__init__()
    self.freq = 50
    self._path = path
    self._loop_times: list[float] = list()

  def loop(self) -> None:
    """Records one timestamp whenever the probe is not paused."""

    self._loop_times.append(time() - self.t0)

  def finish(self) -> None:
    """Writes the collected timestamps after the shared stop event."""

    with self._path.open('w', encoding='utf-8') as file:
      json.dump({'loop_times': self._loop_times,
                 'finish_called': True}, file)


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

  source = ScriptedControlSource()
  pause = crappy.blocks.Pause('pause > 0.5', freq=20)
  stop = crappy.blocks.StopBlock('stop > 0.5', freq=50)
  probe = LifecycleProbe(output_dir / 'pause_probe.json')

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
