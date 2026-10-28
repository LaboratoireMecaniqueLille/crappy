# coding: utf-8

from tests.support.assertions import with_failure_manager

from unittest.mock import Mock, patch
import logging
import re

import numpy as np

import crappy
from crappy.blocks.meta_block.block import VisionBlockType
import crappy.blocks.vision.block as block_module
from crappy.blocks.vision import VisionBlock
from crappy.blocks.vision.block import ConfigRequest, ImgData, ImgLinkData
from crappy.links import ImageLink
from crappy.tool.camera_config.tkinter import TkinterCameraConfig

from tests.support.vision import StubVisionBlock, VisionTestBase


class TestVisionBlockClassAPI(VisionTestBase):
  """Tests VisionBlock construction and small lifecycle helpers."""

  def test_public_alias_and_default_state(self) -> None:
    """Checks the public alias and state established by the constructor."""

    block = StubVisionBlock()

    self.assertIs(crappy.VisionBlock, VisionBlock)
    self.assertIsInstance(block, VisionBlockType)
    self.assertTrue(block.is_vision_block)
    self.assertEqual(block.freq, 200)
    self.assertFalse(block.display_freq)
    self.assertFalse(block.debug)
    self.assertEqual(block.img_inputs, list())
    self.assertEqual(block.img_outputs, list())
    self.assertEqual(block.last_received, dict())
    self.assertIsNone(block.request_config('unused source'))
    self.assertIsNone(block.resolve_config_requests())

  def test_constructor_accepts_valid_image_formats(self) -> None:
    """Checks valid two- and three-dimensional output formats."""

    for shape, dtype in (((2, 3), 'uint8'),
                         ((2, 3, 4), 'float32'),
                         (None, None)):
      with self.subTest(shape=shape, dtype=dtype):
        source = with_failure_manager(StubVisionBlock(img_shape=shape,
                                                     img_dtype=dtype))
        if shape is None:
          source.set_shared_objects()
          source.prepare()
          self.assertEqual(source.img_outputs, [])
          continue
        consumer = with_failure_manager(StubVisionBlock())
        link = ImageLink(source, consumer)
        self.make_manager(source)
        self.set_prepare_sync(consumer)
        source.set_shared_objects()
        source.prepare()
        consumer.prepare()
        image = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        metadata = {'ImageUniqueID': 4, 't(s)': 0.5}

        source.send_img(metadata, image)

        self.assertEqual(consumer.receive_imgs(), [link.name])
        received = consumer.last_received[link.name]
        np.testing.assert_array_equal(received.img, image)
        self.assertEqual(received.img.dtype, image.dtype)
        self.assertEqual(received.metadata, metadata)

  def test_constructor_rejects_invalid_image_format_types(self) -> None:
    """Checks image format container, length, and element types."""

    for shape in ([2, 3], (2,), (2, 3, 4, 5), (2, '3')):
      with self.subTest(shape=shape):
        with self.assertRaises((TypeError, ValueError)):
          with_failure_manager(StubVisionBlock(img_shape=shape))

    for dtype in ('', 1, object()):
      with self.subTest(dtype=dtype):
        with self.assertRaises((TypeError, ValueError)):
          with_failure_manager(StubVisionBlock(img_dtype=dtype))

  def test_constructor_rejects_non_positive_image_dimensions(self) -> None:
    """Checks the documented strictly-positive dimension constraint."""

    for shape in ((0, 3), (-1, 3)):
      with self.subTest(shape=shape):
        with self.assertRaises(ValueError):
          with_failure_manager(StubVisionBlock(img_shape=shape))

  def test_constructor_rejects_unknown_numpy_dtype(self) -> None:
    """Checks dtype names are accepted by NumPy before being stored."""

    with self.assertRaises((TypeError, ValueError)):
      with_failure_manager(StubVisionBlock(img_dtype='not-a-real-numpy-dtype'))

  def test_config_request_validates_boolean_state(self) -> None:
    """Checks mutable ConfigRequest flags and their defaults."""

    request = ConfigRequest(requester='consumer',
                            args=(1,),
                            kwargs={'answer': 42},
                            configurator=TkinterCameraConfig,
                            img_source='source')

    self.assertFalse(request.completed)
    self.assertTrue(request.required)

    for field, value in (('completed', 1), ('required', None)):
      with self.subTest(field=field):
        kwargs = {'completed': False, 'required': True}
        kwargs[field] = value
        with self.assertRaises(TypeError):
          ConfigRequest(requester='consumer',
                        args=tuple(),
                        kwargs=dict(),
                        configurator=TkinterCameraConfig,
                        img_source='source',
                        **kwargs)

  def test_image_link_registration_initializes_receive_state(self) -> None:
    """Checks VisionBlock-specific state installed by an ImageLink."""

    source = with_failure_manager(StubVisionBlock())
    consumer = with_failure_manager(StubVisionBlock())
    link = ImageLink(source, consumer, name='vision-core-image')

    self.assertEqual(source.img_outputs, [link])
    self.assertEqual(consumer.img_inputs, [link])
    self.assertEqual(tuple(consumer.last_received), (link.name,))
    self.assertIsInstance(consumer.last_received[link.name], ImgData)
    self.assertEqual(consumer.last_received[link.name].id, -1)
    self.assertEqual(consumer.last_received[link.name].img.size, 0)

  def test_begin_and_frequency_counter(self) -> None:
    """Checks handled-image frequency timing and counter reset."""

    block = with_failure_manager(StubVisionBlock())

    with patch.object(block_module, 'monotonic', return_value=10.0):
      block.begin()

    block.log = Mock()
    with patch.object(block_module, 'monotonic', return_value=11.0):
      block._print_freq(img_handled=True)
      block._print_freq(img_handled=False)
      block._print_freq(img_handled=True)
    block.log.assert_not_called()

    with patch.object(block_module, 'monotonic', return_value=13.0):
      block._print_freq(img_handled=True)

    self.assertTrue(any(call.args[0] == logging.INFO
                        for call in block.log.call_args_list))
    reported = ' '.join(str(call.args[1]) for call in block.log.call_args_list)
    self.assertIn(1.0, [float(value) for value in
                        re.findall(r'\d+(?:\.\d+)?', reported)])

    block.log.reset_mock()
    with patch.object(block_module, 'monotonic', return_value=14.0):
      block._print_freq(img_handled=False)
    block.log.assert_not_called()
    # A later interval still reports, including when no frame was handled.
    with patch.object(block_module, 'monotonic', return_value=16.0):
      block._print_freq(img_handled=False)
    self.assertTrue(any(call.args[0] == logging.INFO
                        for call in block.log.call_args_list))
    reported = ' '.join(str(call.args[1]) for call in block.log.call_args_list)
    self.assertIn(0.0, [float(value) for value in
                        re.findall(r'\d+(?:\.\d+)?', reported)])

  def test_finish_closes_inputs_and_unlinks_owned_output(self) -> None:
    """Checks ownership rules when releasing shared-memory handles."""

    block = with_failure_manager(StubVisionBlock())
    incoming = Mock()
    outgoing = Mock()
    block._in_link_data = [ImgLinkData(img_buffer=incoming)]
    block._out_link_data = ImgLinkData(img_buffer=outgoing)

    block.finish()

    incoming.close.assert_called_once_with()
    incoming.unlink.assert_not_called()
    outgoing.close.assert_called_once_with()
    outgoing.unlink.assert_called_once_with()


if __name__ == '__main__':
  import unittest
  unittest.main()
