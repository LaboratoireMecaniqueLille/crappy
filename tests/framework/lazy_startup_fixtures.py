"""Importable, hardware-free fixtures for multiprocessing import timing."""

from os import getpid
from time import monotonic_ns, time
from unittest.mock import patch

import numpy as np

from crappy import Block
from crappy.actuator import Actuator
from crappy.blocks import Camera as CameraBlock
from crappy.blocks.camera_processes import CameraProcess
from crappy.blocks.vision import CameraSource, VisionBlock
from crappy.blocks.vision.block import ConfigRequest
from crappy.camera import Camera
from crappy.inout import InOut
from crappy._global import OptionalModule
from crappy.tool.camera_config import CameraConfig
from crappy.tool.camera_config.factory import LazyConfiguratorMapping


def record(events, action):
  events.put((action, getpid(), monotonic_ns()))


class StartupCamera(Camera):
  def open(self, events):
    self.events = events
    record(events, 'camera.open')

  def get_image(self):
    return time(), np.ones((2, 3), dtype='uint8')

  def close(self):
    record(self.events, 'camera.close')


class StartupActuator(Actuator):
  def __init__(self, events):
    super().__init__()
    self.events = events

  def open(self):
    record(self.events, 'actuator.open')

  def set_speed(self, speed):
    pass

  def stop(self):
    pass

  def close(self):
    record(self.events, 'actuator.close')


class StartupInOut(InOut):
  def __init__(self, events):
    super().__init__()
    self.events = events

  def open(self):
    record(self.events, 'inout.open')

  def close(self):
    record(self.events, 'inout.close')


class StartupConfig(CameraConfig):
  """A real concrete configurator class, with no GUI or histogram worker."""

  def __init__(self, camera, *args, **kwargs):
    self.events = camera.events
    self.shape = (2, 3)
    self.dtype = 'uint8'
    record(self.events, 'config.construct')

  def run(self):
    record(self.events, 'config.run')

  def stop(self):
    pass

  def watch_shutdown(self, requested):
    pass

  def get_config(self):
    return ('configured',)


def config_choices():
  return LazyConfiguratorMapping({
    'pyqt': (__name__, 'StartupConfig'),
    'tkinter': ('unused_configuration_backend', 'MustNotImport')})


class StartupSource(CameraSource):
  def __init__(self, events, camera='StartupCamera'):
    super().__init__(camera, config_backend='pyqt', events=events,
                     debug=None)
    self.events = events
    # A downstream request must replace this otherwise unusable default.
    self.configurator = LazyConfiguratorMapping({
      'pyqt': ('unused_default_configuration', 'MustNotImport')})

  def resolve_config_requests(self):
    super().resolve_config_requests()
    record(self.events, 'parent.resolve')

  def prepare(self):
    with (patch('crappy.blocks.vision.camera.resolve_camera',
                side_effect=AssertionError('Driver resolved in worker')),
          patch('crappy.blocks.vision.camera.load_collection_class',
                side_effect=AssertionError('Collection reloaded in worker')),
          patch.object(self, '_resolve_configurator',
                       side_effect=AssertionError('Configurator revalidated in worker')),
          patch('crappy.tool.camera_config.factory.import_module',
                side_effect=AssertionError('Configurator resolved in worker'))):
      super().prepare()
    record(self.events, 'source.prepared')


class StartupConsumer(VisionBlock):
  def __init__(self, events):
    super().__init__(debug=None)
    self.events = events
    self.configurator = config_choices()

  def request_config(self, source):
    return ConfigRequest(self.name, (), {}, self.configurator, source)

  def prepare(self):
    assert tuple(self.recv_configs().values()) == (('configured',),)
    super().prepare()

  def loop(self):
    if self.receive_imgs():
      record(self.events, 'consumer.frame')
      self.stop()


class StartupProcess(CameraProcess):
  def __init__(self, events):
    super().__init__()
    self.events = events
    self.backend_name = 'json'
    self._guard = None

  def init(self):
    backend = OptionalModule(self.backend_name, lazy_import=True)
    self._decode = backend.resolve().loads
    record(self.events, 'process.init')
    self._guard = patch('crappy._global.import_module',
                        side_effect=AssertionError('Resolution in frame loop'))
    self._guard.start()

  def loop(self):
    assert self._decode('1') == 1
    assert self.img.shape == (2, 3)
    record(self.events, 'process.frame')
    self._stop_event.set()

  def finish(self):
    if self._guard is not None:
      self._guard.stop()
    record(self.events, 'process.finish')


class StartupLegacyCamera(CameraBlock):
  configurator = config_choices()

  def __init__(self, events, camera='StartupCamera'):
    super().__init__(camera, config_backend='pyqt', events=events,
                     debug=None)
    self.events = events
    record(self.events, 'parent.resolve')

  def prepare(self):
    self.process_proc = StartupProcess(self.events)
    with (patch('crappy.blocks.camera.resolve_camera',
                side_effect=AssertionError('Driver resolved in worker')),
          patch('crappy.blocks.camera.load_collection_class',
                side_effect=AssertionError('Collection reloaded in worker')),
          patch('crappy.tool.camera_config.factory.import_module',
                side_effect=AssertionError('Configurator resolved in worker'))):
      super().prepare()


def register_collection_drivers():
  """Declare real importable collection classes without loading their module."""

  import sys
  from crappy._collection import CollectionEntry, collection_registry

  module = 'tests.collection.lazy_driver_fixtures'
  assert module not in sys.modules
  collection_registry.register(*(
    CollectionEntry('CollectionStartup' + kind, kind, module)
    for kind in ('Actuator', 'Camera', 'InOut')))


def run_pipeline(method, legacy, collection=False):
  """Run a complete camera pipeline in a fresh interpreter/start context."""

  import logging
  import multiprocessing as mp
  from queue import Empty
  import warnings
  from crappy import img_link

  mp.set_start_method(method, force=True)
  warnings.simplefilter('ignore')
  events = mp.Queue()
  if collection:
    register_collection_drivers()
  camera = 'CollectionStartupCamera' if collection else 'StartupCamera'
  source_type = StartupLegacyCamera if legacy else StartupSource
  source = source_type(events, camera)
  selected_camera = source._camera_class
  assert selected_camera.__name__ == camera
  if not legacy:
    consumer = StartupConsumer(events)
    img_link(source, consumer)
  try:
    Block.prepare_all(log_level=logging.CRITICAL)
    assert source._camera_class is selected_camera
    if legacy:
      assert source.configurator is StartupConfig
    else:
      assert source.config_requests_in[0].configurator is StartupConfig
    Block.launch_all()
    reports = []
    while True:
      try:
        reports.append(events.get(timeout=0.2))
      except Empty:
        break
    actions = {action: (pid, timestamp) for action, pid, timestamp in reports}
    parent_pid, resolved = actions['parent.resolve']
    assert parent_pid == getpid()
    for action in ('camera.open', 'config.construct', 'config.run', 'camera.close'):
      pid, timestamp = actions[action]
      assert pid != parent_pid and timestamp > resolved, reports
    if legacy:
      init_pid, initialized = actions['process.init']
      frame_pid, processed = actions['process.frame']
      assert init_pid == frame_pid != actions['camera.open'][0]
      assert resolved < initialized < processed
      assert 'process.finish' in actions
    else:
      assert actions['consumer.frame'][1] > actions['source.prepared'][1]
    assert not mp.active_children(), mp.active_children()
  finally:
    Block.reset()
    events.close()
    events.join_thread()


def prepare_scalar_owner(block, events):
  """A fresh worker must use the real driver selected by its parent."""

  from crappy._failure import FailureManager
  from unittest.mock import Mock

  block._failure_manager = FailureManager(block.name, 'prepare')
  block._failure_manager._logger = Mock()
  block.log = Mock()
  with (patch('crappy.blocks.machine.resolve_actuator', side_effect=AssertionError),
        patch('crappy.blocks.machine.load_collection_class', side_effect=AssertionError),
        patch('crappy.blocks.auto_drive_video_extenso.resolve_actuator',
              side_effect=AssertionError),
        patch('crappy.blocks.auto_drive_video_extenso.load_collection_class',
              side_effect=AssertionError),
        patch('crappy.blocks.ioblock.resolve_inout', side_effect=AssertionError),
        patch('crappy.blocks.ioblock.load_collection_class', side_effect=AssertionError)):
    try:
      block.prepare()
      record(events, 'driver.prepared')
    finally:
      block.finish()


def run_scalar_owners(method, collection=False):
  import multiprocessing as mp
  from crappy.blocks import AutoDriveVideoExtenso, IOBlock, Machine

  mp.set_start_method(method, force=True)
  events = mp.Queue()
  if collection:
    register_collection_drivers()
  actuator = 'CollectionStartupActuator' if collection else 'StartupActuator'
  inout = 'CollectionStartupInOut' if collection else 'StartupInOut'
  owners = (Machine([{'type': actuator, 'events': events}], debug=None),
            AutoDriveVideoExtenso({'type': actuator, 'events': events},
                                  debug=None),
            IOBlock(inout, cmd_labels=['cmd'], events=events, debug=None))
  try:
    for owner in owners:
      owner.inputs.append('linked')
      process = mp.Process(target=prepare_scalar_owner, args=(owner, events))
      try:
        process.start()
        reports = [events.get(timeout=10) for _ in range(3)]
        process.join(timeout=10)
        assert process.exitcode == 0, (method, type(owner), process.exitcode)
        assert reports[1][0] == 'driver.prepared', reports
        assert all(pid != getpid() for _, pid, _ in reports), reports
      finally:
        if process.is_alive():
          process.terminate()
          process.join(timeout=5)
  finally:
    Block.reset()
    events.close()
    events.join_thread()
