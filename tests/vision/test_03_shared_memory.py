# coding: utf-8

from multiprocessing import Barrier, Event, RLock, Value
from unittest.mock import Mock

import numpy as np

from crappy._global import LinkDataError, PrepareError
from crappy.blocks.vision.block import ImgLinkData
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
    self.assertIs(lock, source._out_link_data.img_lock)
    self.assertIs(metadata, source._out_link_data.metadata_dict)
    self.assertIs(ready, source._out_link_data.buffer_ready)
    self.assertIs(info, source._out_link_data.img_info_dict)
    self.assertIs(img_id, source._out_link_data.img_id)
    self.assertEqual(img_id.value, -1)
    self.assertFalse(ready.is_set())

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
      block._get_image_buffer('unused', ready, {})

    waiting = Mock()
    waiting.wait.return_value = False
    block._stop_event.set()
    with self.assertRaises(PrepareError):
      block._get_image_buffer('unused', waiting,
                              {'shape': (2, 2), 'dtype': 'uint8'})
    waiting.wait.assert_called_once_with(0.5)

    block._ready_barrier = None
    with self.assertRaises(ValueError):
      block._get_image_buffer('unused', waiting,
                              {'shape': (2, 2), 'dtype': 'uint8'})

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
