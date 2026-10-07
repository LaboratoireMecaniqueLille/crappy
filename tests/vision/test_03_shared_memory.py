# coding: utf-8

from multiprocessing import Barrier, Event, RLock, Value
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np

from crappy._global import LinkDataError, PrepareError
from crappy.blocks.vision.block import ImgLinkData
import crappy.blocks.vision.block as block_module
from crappy.links import ImageLink

from .vision_test_base import StubVisionBlock, VisionTestBase


class TestVisionBlockSharedMemory(VisionTestBase):
  """Tests VisionBlock shared-memory setup and frame exchange."""

  def prepare_pair(self,
                   shape=(2, 3),
                   dtype='uint16'
                   ) -> tuple[StubVisionBlock, StubVisionBlock, ImageLink]:
    """Creates and prepares a source/consumer pair with real shared memory."""

    source = StubVisionBlock(img_shape=shape, img_dtype=dtype)
    consumer = StubVisionBlock()
    link = ImageLink(source, consumer, name='prepared-image')
    self.make_manager(source)
    self.set_prepare_sync(consumer)

    source.set_shared_objects()
    source.prepare()
    consumer.prepare()
    return source, consumer, link

  def test_set_shared_objects_is_noop_without_outputs(self) -> None:
    """Checks that sink VisionBlocks do not require a Manager."""

    block = StubVisionBlock()

    block.set_shared_objects()

    self.assertEqual(block._out_link_data, ImgLinkData())

  def test_set_shared_objects_requires_manager_and_populates_link(self) -> None:
    """Checks publication of a complete synchronization-object bundle."""

    source = StubVisionBlock(img_shape=(2, 3), img_dtype='uint8')
    consumer = StubVisionBlock()
    link = ImageLink(source, consumer, name='shared-objects')

    with self.assertRaises(ValueError):
      source.set_shared_objects()

    self.make_manager(source)
    source.set_shared_objects()
    buffers = link.get_buffers()

    self.assertIsNotNone(buffers)
    name, lock, metadata, ready, info, img_id = buffers
    self.assertEqual(name, source._out_link_data.memory_name)
    self.assertRegex(name, r'^crappy_[A-Za-z0-9_-]{22}$')
    self.assertLessEqual(len(f'/{name}'.encode('ascii')), 31)
    self.assertIs(lock, source._out_link_data.img_lock)
    self.assertIs(metadata, source._out_link_data.metadata_dict)
    self.assertIs(ready, source._out_link_data.buffer_ready)
    self.assertIs(info, source._out_link_data.img_info_dict)
    self.assertIs(img_id, source._out_link_data.img_id)
    self.assertEqual(img_id.value, -1)
    self.assertFalse(ready.is_set())

  def test_shared_memory_name_does_not_depend_on_block_or_link_length(
      self) -> None:
    """Checks long user-facing names cannot exceed the macOS SHM limit."""

    source = StubVisionBlock(img_shape=(2, 3), img_dtype='uint8')
    source.name = 'source-' + 'x' * 100
    consumer = StubVisionBlock()
    link = ImageLink(source, consumer, name='images-' + 'y' * 100)
    self.make_manager(source)
    source.log = Mock()

    source.set_shared_objects()
    source.prepare()

    name, *_ = link.get_buffers()
    self.assertRegex(name, r'^crappy_[A-Za-z0-9_-]{22}$')
    self.assertLessEqual(len(f'/{name}'.encode('ascii')), 31)
    self.assertNotIn(source.name, name)
    self.assertNotIn(link.name, name)
    messages = [call.args[1] for call in source.log.call_args_list]
    self.assertTrue(any(name in message and link.name in message
                        for message in messages))

  def test_prepare_validates_output_format_and_shared_state(self) -> None:
    """Checks failures before allocation of an invalid output buffer."""

    block = StubVisionBlock()
    block.img_outputs.append(object())

    with self.assertRaises(ValueError):
      block.prepare()

    for shape, dtype in (([2, 3], 'uint8'),
                         ((2,), 'uint8'),
                         ((2, 3), np.uint8)):
      with self.subTest(shape=shape, dtype=dtype):
        block._img_shape = shape
        block._img_dtype = dtype
        with self.assertRaises(ValueError):
          block.prepare()

    block._img_shape = (2, 3)
    block._img_dtype = 'uint8'
    with self.assertRaises(ValueError):
      block.prepare()

  def test_real_shared_memory_round_trip_and_latest_frame(self) -> None:
    """Checks atomic metadata/image copies and latest-frame semantics."""

    source, consumer, link = self.prepare_pair()
    first = np.arange(6, dtype=np.uint16).reshape(2, 3)
    metadata = {'ImageUniqueID': 10, 't(s)': 0.1, 'camera': 'fake'}

    source.send_img(metadata, first)

    self.assertEqual(consumer.receive_imgs(), [link.name])
    received = consumer.last_received[link.name]
    self.assertEqual(received.id, 0)
    self.assertEqual(received.metadata, metadata)
    self.assertIsNot(received.metadata, metadata)
    np.testing.assert_array_equal(received.img, first)
    self.assertEqual(consumer.receive_imgs(), list())

    second = first + 10
    third = first + 20
    source.send_img({'ImageUniqueID': 11, 't(s)': 0.2}, second)
    source.send_img({'ImageUniqueID': 12, 't(s)': 0.3}, third)

    self.assertEqual(consumer.receive_imgs(), [link.name])
    self.assertEqual(received.id, 2)
    self.assertEqual(received.metadata['ImageUniqueID'], 12)
    np.testing.assert_array_equal(received.img, third)
    self.assertEqual(source._sent_img_counter, 3)

    # The consumer owns a local copy, not a view over the shared source buffer.
    source._out_link_data.npy_buffer[:] = 0
    np.testing.assert_array_equal(received.img, third)

  def test_one_source_buffer_can_feed_multiple_consumers(self) -> None:
    """Checks that every outgoing ImageLink receives the same buffer bundle."""

    source = StubVisionBlock(img_shape=(2, 2), img_dtype='uint8')
    first = StubVisionBlock()
    second = StubVisionBlock()
    first_link = ImageLink(source, first, name='first-consumer')
    second_link = ImageLink(source, second, name='second-consumer')
    self.make_manager(source)
    self.set_prepare_sync(first)
    self.set_prepare_sync(second)
    source.set_shared_objects()
    source.prepare()
    first.prepare()
    second.prepare()

    image = np.arange(4, dtype=np.uint8).reshape(2, 2)
    metadata = {'ImageUniqueID': 1, 't(s)': 0.5}
    source.send_img(metadata, image)

    self.assertEqual(first.receive_imgs(), [first_link.name])
    self.assertEqual(second.receive_imgs(), [second_link.name])
    np.testing.assert_array_equal(first.last_received[first_link.name].img,
                                  image)
    np.testing.assert_array_equal(second.last_received[second_link.name].img,
                                  image)

  def test_send_img_validates_data_and_prepared_format(self) -> None:
    """Checks types, mandatory metadata, shape, and dtype before copying."""

    source, _, _ = self.prepare_pair(shape=(2, 2), dtype='uint8')
    image = np.zeros((2, 2), dtype=np.uint8)
    metadata = {'ImageUniqueID': 1, 't(s)': 0.1}

    with self.assertRaises(LinkDataError):
      source.send_img([], image)
    with self.assertRaises(LinkDataError):
      source.send_img(metadata, image.tolist())

    for invalid_metadata in ({'t(s)': 0.1}, {'ImageUniqueID': 1}):
      with self.subTest(metadata=invalid_metadata):
        with self.assertRaises(ValueError):
          source.send_img(invalid_metadata, image)

    with self.assertRaises(ValueError):
      source.send_img(metadata, image.astype(np.uint16))
    with self.assertRaises(ValueError):
      source.send_img(metadata, np.zeros((1, 4), dtype=np.uint8))

    unprepared = StubVisionBlock()
    with self.assertRaises(ValueError):
      unprepared.send_img(metadata, image)

  def test_receive_imgs_detects_inconsistent_buffers(self) -> None:
    """Checks consumer-side initialization and dtype/shape validation."""

    consumer = StubVisionBlock()
    link = Mock(name='image-link')
    link.name = 'manual-image'
    consumer.add_img_input(link)
    consumer._in_link_data = [ImgLinkData(img_lock=RLock(),
                                          metadata_dict={
                                            'ImageUniqueID': 1,
                                            't(s)': 0.1,
                                          },
                                          img_id=Value('l', 0),
                                          npy_buffer=np.zeros((2, 2),
                                                              dtype=np.uint8))]

    with self.assertRaises(ValueError):
      consumer.receive_imgs()

    consumer.last_received[link.name].img = np.zeros((2, 2),
                                                      dtype=np.uint16)
    with self.assertRaises(ValueError):
      consumer.receive_imgs()

    consumer.last_received[link.name].img = np.zeros((1, 4), dtype=np.uint8)
    with self.assertRaises(ValueError):
      consumer.receive_imgs()

  def test_get_image_buffer_validates_info_and_notices_abort(self) -> None:
    """Checks attachment validation and interruption while waiting."""

    block = StubVisionBlock()
    self.set_prepare_sync(block)
    ready = Event()
    ready.set()

    with self.assertRaises(ValueError):
      block._get_image_buffer(ImgLinkData(memory_name='unused',
                                          buffer_ready=ready,
                                          img_info_dict={}))

    waiting = Mock()
    waiting.wait.return_value = False
    block._stop_event.set()
    with self.assertRaises(PrepareError):
      block._get_image_buffer(ImgLinkData(
          memory_name='unused', buffer_ready=waiting,
          img_info_dict={'shape': (2, 2), 'dtype': 'uint8'}))
    waiting.wait.assert_called_once_with(0.5)

    block._ready_barrier = None
    with self.assertRaises(ValueError):
      block._get_image_buffer(ImgLinkData(
          memory_name='unused', buffer_ready=waiting,
          img_info_dict={'shape': (2, 2), 'dtype': 'uint8'}))

  def test_finish_releases_output_after_array_creation_failure(self) -> None:
    """Allocation remains owned even if the Numpy view cannot be created."""

    source = StubVisionBlock(img_shape=(2, 3), img_dtype='uint8')
    ImageLink(source, StubVisionBlock(), name='failed-output-view')
    self.make_manager(source)
    source.set_shared_objects()

    with patch.object(block_module.np, 'ndarray',
                      side_effect=ValueError('cannot create image view')):
      with self.assertRaises(ValueError):
        source.prepare()

    handle = source._out_link_data.img_buffer
    self.assertIsNotNone(handle)
    self.assertIsNone(source._out_link_data.npy_buffer)
    source.finish()
    self.assertIsNone(handle.buf)
    with self.assertRaises(FileNotFoundError):
      SharedMemory(name=handle.name)
    source.finish()

  def test_finish_closes_attachment_after_array_creation_failure(self) -> None:
    """A failed view must not leak an attached handle or unlink its source."""

    source, consumer, _ = self.prepare_pair(shape=(2, 2), dtype='uint8')
    consumer.finish()
    data = consumer._in_link_data[0]
    data.img_info_dict['shape'] = (200, 200)

    with self.assertRaises(TypeError):
      consumer._get_image_buffer(data)
    handle = data.img_buffer
    self.assertIsNotNone(handle)
    consumer.finish()
    self.assertIsNone(handle.buf)
    self.assertIsNone(data.img_buffer)
    attachment = SharedMemory(name=source._out_link_data.memory_name)
    attachment.close()

  def test_finish_attempts_all_memory_operations_and_notifications(self) -> None:
    """Failed close/unlink cannot prevent other buffers and readers cleanup."""

    block = StubVisionBlock()
    operations = Mock()
    first, second, output = Mock(), Mock(), Mock()
    for name, handle in (('first', first), ('second', second),
                         ('output', output)):
      operations.attach_mock(handle, name)
    errors = [RuntimeError('second close'), RuntimeError('output close'),
              RuntimeError('output unlink'), RuntimeError('notify')]
    second.close.side_effect = errors[0]
    output.close.side_effect = errors[1]
    output.unlink.side_effect = errors[2]
    block._in_link_data = [ImgLinkData(memory_name='first', img_buffer=first),
                           ImgLinkData(memory_name='second', img_buffer=second)]
    block._out_link_data = ImgLinkData(memory_name='output', img_buffer=output)
    condition = MagicMock()
    condition.notify_all.side_effect = errors[3]
    block._out_img_conditions = [condition, MagicMock()]

    with self.assertRaises(ExceptionGroup) as caught:
      block.finish()

    self.assertEqual(caught.exception.exceptions, tuple(errors))
    for error in errors:
      self.assertTrue(error.__notes__[0].startswith('VisionBlock cleanup step:'))
    self.assertEqual(operations.mock_calls,
                     [call.second.close(), call.first.close(),
                      call.output.close(), call.output.unlink()])
    block._out_img_conditions[1].notify_all.assert_called_once_with()
    first.unlink.assert_not_called()
    second.unlink.assert_not_called()

    second.close.side_effect = None
    output.close.side_effect = None
    output.unlink.side_effect = None
    condition.notify_all.side_effect = None
    block.finish()
    block.finish()
    self.assertEqual(first.close.call_count, 1)
    self.assertEqual(second.close.call_count, 2)
    self.assertEqual(output.close.call_count, 2)
    self.assertEqual(output.unlink.call_count, 2)

  def test_finish_retries_only_failed_output_operation(self) -> None:
    """Successful close and successful unlink are independently remembered."""

    for failed in ('close', 'unlink'):
      with self.subTest(failed=failed):
        block = StubVisionBlock()
        handle = Mock()
        error = RuntimeError(failed)
        getattr(handle, failed).side_effect = error
        block._out_link_data = ImgLinkData(img_buffer=handle)
        with self.assertRaises(RuntimeError) as caught:
          block.finish()
        self.assertIs(caught.exception, error)
        getattr(handle, failed).side_effect = None
        block.finish()
        block.finish()
        self.assertEqual(handle.close.call_count, 2 if failed == 'close' else 1)
        self.assertEqual(handle.unlink.call_count, 2 if failed == 'unlink' else 1)

  def test_finish_accepts_already_unlinked_output(self) -> None:
    """Another owner may already have removed the output segment."""

    block = StubVisionBlock()
    handle = Mock()
    handle.unlink.side_effect = FileNotFoundError
    block._out_link_data = ImgLinkData(img_buffer=handle)
    block.finish()
    block.finish()
    handle.close.assert_called_once_with()
    handle.unlink.assert_called_once_with()
    self.assertIsNone(block._out_link_data.img_buffer)

  def test_finish_prioritizes_interrupt_after_releasing_other_buffers(self
                                                                    ) -> None:
    """The original interrupt keeps the other errors in its cause."""

    block = StubVisionBlock()
    input_handle, output_handle = Mock(), Mock()
    interrupt = KeyboardInterrupt('input close interrupted')
    error = RuntimeError('output close failed')
    input_handle.close.side_effect = interrupt
    output_handle.close.side_effect = error
    block._in_link_data = [ImgLinkData(img_buffer=input_handle)]
    block._out_link_data = ImgLinkData(img_buffer=output_handle)
    with self.assertRaises(KeyboardInterrupt) as caught:
      block.finish()
    self.assertIs(caught.exception, interrupt)
    self.assertEqual(interrupt.__cause__.exceptions, (error,))
    output_handle.unlink.assert_called_once_with()
    input_handle.close.side_effect = None
    output_handle.close.side_effect = None
    block.finish()

  def test_get_shared_objects_rejects_incomplete_image_link(self) -> None:
    """Checks the error when an input ImageLink has no buffer bundle."""

    block = StubVisionBlock()
    link = Mock()
    link.name = 'incomplete-image'
    link.get_buffers.return_value = None
    block.add_img_input(link)

    with self.assertRaises(RuntimeError):
      block._get_shared_objects()


if __name__ == '__main__':
  import unittest
  unittest.main()
