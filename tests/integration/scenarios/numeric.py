# coding: utf-8

from pathlib import Path

import crappy
from crappy.blocks import Block


def build_generator_recorder(output_dir: Path) -> tuple[Block, ...]:
  """Builds a finite Generator -> Recorder script."""

  generator = crappy.blocks.Generator(
    ({'type': 'Ramp',
      'speed': 2,
      'condition': 'delay=0.6',
      'init_value': 1},),
    cmd_label='signal',
    spam=True,
    end_delay=0.2,
    freq=50)

  recorder = crappy.blocks.Recorder(output_dir / 'signal.csv',
                                    labels=('t(s)', 'signal'),
                                    delay=0.17,
                                    freq=20)

  crappy.link(generator, recorder)

  return generator, recorder


def build_multiplexer_recorder(output_dir: Path) -> tuple[Block, ...]:
  """Builds a two Generators -> Multiplexer -> Recorder script."""

  first_generator = crappy.blocks.Generator(
    ({'type': 'Ramp',
      'speed': 2,
      'condition': 'delay=0.7',
      'init_value': 1},),
    cmd_label='first',
    spam=True,
    end_delay=0.2,
    freq=40)

  second_generator = crappy.blocks.Generator(
    ({'type': 'Ramp',
      'speed': -3,
      'condition': 'delay=0.7',
      'init_value': 5},),
    cmd_label='second',
    spam=True,
    end_delay=0.2,
    freq=25)

  multiplexer = crappy.blocks.Multiplexer(
    out_labels=('first', 'second'),
    interp_freq=20,
    freq=50)

  recorder = crappy.blocks.Recorder(
    output_dir / 'multiplexed.csv',
    labels=('t(s)', 'first', 'second'),
    delay=0.1,
    freq=20)

  crappy.link(first_generator, multiplexer)
  crappy.link(second_generator, multiplexer)
  crappy.link(multiplexer, recorder)

  return first_generator, second_generator, multiplexer, recorder


def build_synchronizer_recorder(output_dir: Path) -> tuple[Block, ...]:
  """Builds a two Generators -> Synchronizer -> Recorder script."""

  reference_generator = crappy.blocks.Generator(
    ({'type': 'Ramp',
      'speed': 3,
      'condition': 'delay=0.7',
      'init_value': 10},),
    cmd_label='reference',
    spam=True,
    end_delay=0.2,
    freq=35)

  signal_generator = crappy.blocks.Generator(
    ({'type': 'Ramp',
      'speed': 4,
      'condition': 'delay=0.7',
      'init_value': -2},),
    cmd_label='signal',
    spam=True,
    end_delay=0.2,
    freq=23)

  synchronizer = crappy.blocks.Synchronizer(
    reference_label='reference',
    labels_to_sync='signal',
    freq=50)

  recorder = crappy.blocks.Recorder(
    output_dir / 'synchronized.csv',
    labels=('t(s)', 'reference', 'signal'),
    delay=0.1,
    freq=20)

  crappy.link(reference_generator, synchronizer)
  crappy.link(signal_generator, synchronizer)
  crappy.link(synchronizer, recorder)

  return reference_generator, signal_generator, synchronizer, recorder


def build_pid_fake_machine_recorder(output_dir: Path) -> tuple[Block, ...]:
  """Builds a Generator/feedback -> PID -> FakeMachine -> Recorder script."""

  generator = crappy.blocks.Generator(
    ({'type': 'Constant',
      'value': 0.4,
      'condition': 'delay=0.8'},),
    cmd_label='target_position',
    spam=True,
    end_delay=0.2,
    freq=40)

  pid = crappy.blocks.PID(
    kp=5,
    ki=0.5,
    out_min=-1.5,
    out_max=1.5,
    setpoint_label='target_position',
    input_label='x(mm)',
    labels=('t(s)', 'machine_command'),
    freq=50)

  machine = crappy.blocks.FakeMachine(
    mode='speed',
    cmd_label='machine_command',
    max_speed=1.5,
    sigma={},
    freq=50)

  recorder = crappy.blocks.Recorder(
    output_dir / 'pid_machine.csv',
    labels=('t(s)', 'x(mm)', 'F(N)'),
    delay=0.1,
    freq=20)

  crappy.link(generator, pid)
  crappy.link(machine, pid)
  crappy.link(pid, machine)
  crappy.link(machine, recorder)

  return generator, pid, machine, recorder


def build_machine_recorder(output_dir: Path) -> tuple[Block, ...]:
  """Builds a Generator -> Machine(FakeDCMotor) -> Recorder script."""

  generator = crappy.blocks.Generator(
    ({'type': 'Constant',
      'value': 1.5,
      'condition': 'delay=0.7'},),
    cmd_label='voltage',
    spam=True,
    end_delay=0.2,
    freq=40)

  machine = crappy.blocks.Machine(
    ({'type': 'FakeDCMotor',
      'cmd_label': 'voltage',
      'mode': 'speed',
      'speed_label': 'motor_speed',
      'position_label': 'motor_position',
      'inertia': 2,
      'kv': 100,
      'rv': 0.2,
      'fv': 1e-5},),
    freq=50)

  recorder = crappy.blocks.Recorder(
    output_dir / 'motor.csv',
    labels=('t(s)', 'motor_speed', 'motor_position'),
    delay=0.1,
    freq=20)

  crappy.link(generator, machine)
  crappy.link(machine, recorder)

  return generator, machine, recorder


def build_ioblock_recorder(output_dir: Path) -> tuple[Block, ...]:
  """Builds an IOBlock(FakeInOut) -> Recorder script."""

  ioblock = crappy.blocks.IOBlock(
    'FakeInOut',
    labels=('t(s)', 'memory'),
    freq=30)

  recorder = crappy.blocks.Recorder(
    output_dir / 'memory.csv',
    labels=('t(s)', 'memory'),
    delay=0.1,
    freq=20)

  stop = crappy.blocks.StopBlock('t(s) > 0.7', freq=50)

  crappy.link(ioblock, recorder)

  return ioblock, recorder, stop


def build_stream_hdf_recorder(output_dir: Path) -> tuple[Block, ...]:
  """Builds a FakeInOut stream source -> HDFRecorder script."""

  stream = crappy.blocks.IOBlock(
    'FakeInOut',
    labels=('t(s)', 'stream'),
    streamer=True,
    freq=30)

  recorder = crappy.blocks.HDFRecorder(
    output_dir / 'stream.h5',
    atom='float64',
    label='stream',
    expected_rows=500,
    flush_period=2,
    freq=100)

  stop = crappy.blocks.StopBlock('t(s) > 0.7', freq=50)

  crappy.link(stream, recorder)

  return stream, recorder, stop
