# coding: utf-8

from multiprocessing import Barrier, Event, Manager
from multiprocessing.connection import Connection
from multiprocessing.managers import SyncManager
from multiprocessing.shared_memory import SharedMemory

from crappy import Block
from crappy.blocks.vision import CameraSource, VisionBlock

from tests.support.block_test_case import BlockTestBase
from tests.support.assertions import with_failure_manager
from tests.support.processes import stop_process
from tests.support.resources import preserve_mapping, run_cleanups


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
    self._first_stub = len(StubVisionBlock.test_instances)
    preserve_mapping(self, CameraSource.cam_count)
    self.addCleanup(self._cleanup_vision)

  def track_block(self, block: VisionBlock) -> VisionBlock:
    """Keeps a strong reference to a concrete VisionBlock until cleanup."""

    with_failure_manager(block)
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
    """Release independent handles without hiding a failed close."""

    def release(data, owner):
      data.npy_buffer = None
      handle = data.img_buffer
      try:
        # Fault-injection doubles do not own real shared-memory handles.
        if isinstance(handle, SharedMemory):
          def unlink():
            try:
              handle.unlink()
            except FileNotFoundError:
              pass  # Production shutdown may already have unlinked it.
          run_cleanups(handle.close, *([unlink] if owner else []))
      finally:
        data.img_buffer = None

    inputs = getattr(block, '_in_link_data', ())
    output = getattr(block, '_out_link_data', None)
    run_cleanups(*(lambda data=data: release(data, False) for data in inputs),
                 *([lambda: release(output, True)] if output is not None else []))

  def _cleanup_vision(self) -> None:
    """Releases shared memory, Pipe endpoints, and Manager processes."""

    try:
      blocks = [*tuple(Block.instances),
                *StubVisionBlock.test_instances[self._first_stub:],
                *self._vision_blocks]
      blocks = [block for block in dict.fromkeys(blocks)
                if isinstance(block, VisionBlock)]

      def stop(block):
        def request():
          if block._stop_event is not None:
            block._stop_event.set()
        stop_process(block, request)

      run_cleanups(*(lambda block=block: stop(block) for block in blocks),
                   *(lambda block=block: self._release_block_memory(block)
                     for block in blocks),
                   *(connection.close for connection in self._connections),
                   *(manager.shutdown for manager in self._managers))

    finally:
      del StubVisionBlock.test_instances[self._first_stub:]
