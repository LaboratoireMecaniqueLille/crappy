# coding: utf-8

from multiprocessing import Barrier, Event, Manager
from multiprocessing.connection import Connection
from multiprocessing.managers import SyncManager
from multiprocessing.shared_memory import SharedMemory

from crappy import Block
from crappy.blocks.vision import CameraSource, VisionBlock

from tests.block.block_test_base import BlockTestBase


class StubVisionBlock(VisionBlock):
  """Small concrete VisionBlock used by the core unit tests."""

  test_instances: list['StubVisionBlock'] = list()

  def __init__(self, *args, **kwargs) -> None:
    """Keeps a strong test-only reference for shared-memory cleanup."""

    super().__init__(*args, **kwargs)
    type(self).test_instances.append(self)

  def loop(self) -> None:
    """Provide a no-op loop implementation for lifecycle compatibility."""

    ...


class VisionTestBase(BlockTestBase):
  """Common resource management helpers for VisionBlock unit tests."""

  def setUp(self) -> None:
    """Initializes resource registries in addition to Block isolation."""

    super().setUp()
    self._managers: list[SyncManager] = list()
    self._connections: list[Connection] = list()
    self._vision_blocks: list[VisionBlock] = list()

  def track_block(self, block: VisionBlock) -> VisionBlock:
    """Keeps a strong reference to a concrete VisionBlock until cleanup."""

    self._vision_blocks.append(block)
    return block

  def make_manager(self, *blocks: VisionBlock) -> SyncManager:
    """Creates a Manager and exposes it to the provided VisionBlocks."""

    manager = Manager()
    self._managers.append(manager)
    for block in blocks:
      block.shared_mgr = manager
    return manager

  @staticmethod
  def set_prepare_sync(block: VisionBlock) -> None:
    """Installs the synchronization objects used during preparation."""

    block._ready_barrier = Barrier(1)
    block._stop_event = Event()

  def track_connection(self, connection: Connection) -> Connection:
    """Registers a Pipe endpoint for cleanup and returns it."""

    self._connections.append(connection)
    return connection

  @staticmethod
  def _release_block_memory(block: VisionBlock) -> None:
    """Best-effort cleanup for shared-memory handles left by a test."""

    for data in getattr(block, '_in_link_data', tuple()):
      data.npy_buffer = None
      if data.img_buffer is not None:
        try:
          data.img_buffer.close()
        except (BufferError, OSError):
          pass
        data.img_buffer = None

    data = getattr(block, '_out_link_data', None)
    if data is not None and data.img_buffer is not None:
      data.npy_buffer = None
      try:
        data.img_buffer.close()
      except (BufferError, OSError):
        pass
      try:
        data.img_buffer.unlink()
      except (FileNotFoundError, OSError):
        pass
      data.img_buffer = None

  def tearDown(self) -> None:
    """Releases shared memory, Pipe endpoints, and Manager processes."""

    try:
      blocks = [*tuple(Block.instances),
                *StubVisionBlock.test_instances,
                *self._vision_blocks]
      for block in dict.fromkeys(blocks):
        if isinstance(block, VisionBlock):
          self._release_block_memory(block)

      StubVisionBlock.test_instances.clear()
      CameraSource.cam_count.clear()

      for connection in self._connections:
        try:
          connection.close()
        except OSError:
          pass

      for manager in self._managers:
        manager.shutdown()

    finally:
      super().tearDown()
