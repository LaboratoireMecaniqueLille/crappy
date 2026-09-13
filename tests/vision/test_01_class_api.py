# coding: utf-8

from unittest.mock import Mock, patch
import logging

import numpy as np

import crappy
import crappy.blocks.vision.block as block_module
from crappy.blocks.vision import VisionBlock
from crappy.blocks.vision.block import ConfigRequest, ImgData, ImgLinkData
from crappy.links import ImageLink
from crappy.tool.camera_config import CameraConfig

from .vision_test_base import StubVisionBlock, VisionTestBase


class TestVisionBlockClassAPI(VisionTestBase):
  """Tests VisionBlock construction and small lifecycle helpers."""

  def test_public_alias_and_default_state(self) -> None:
    """Checks the public alias and state established by the constructor."""

    block = StubVisionBlock()

    self.assertIs(crappy.VisionBlock, VisionBlock)
    self.assertTrue(block.is_vision_block)
    self.assertEqual(block.freq, 200)
    self.assertFalse(block.display_freq)
    self.assertFalse(block.debug)
    self.assertEqual(block.img_inputs, list())
    self.assertEqual(block.img_outputs, list())
    self.assertEqual(block.last_received, dict())
    self.assertIsInstance(block._out_link_data, ImgLinkData)
    self.assertEqual(block._sent_img_counter, 0)
    self.assertIsNone(block.request_config('unused source'))

  def test_constructor_accepts_valid_image_formats(self) -> None:
    """Checks valid two- and three-dimensional output formats."""

    for shape, dtype in (((2, 3), 'uint8'),
                         ((2, 3, 4), 'float32'),
                         (None, None)):
      with self.subTest(shape=shape, dtype=dtype):
        block = StubVisionBlock(img_shape=shape, img_dtype=dtype)
        self.assertEqual(block._img_shape, shape)
        self.assertEqual(block._img_dtype, dtype)

  def test_constructor_rejects_invalid_image_format_types(self) -> None:
    """Checks image format container, length, and element types."""

    for shape in ([2, 3], (2,), (2, 3, 4, 5), (2, '3')):
      with self.subTest(shape=shape):
        with self.assertRaises((TypeError, ValueError)):
          StubVisionBlock(img_shape=shape)

    for dtype in ('', 1, object()):
      with self.subTest(dtype=dtype):
        with self.assertRaises((TypeError, ValueError)):
          StubVisionBlock(img_dtype=dtype)

  def test_constructor_rejects_non_positive_image_dimensions(self) -> None:
    """Checks the documented strictly-positive dimension constraint."""

    for shape in ((0, 3), (-1, 3)):
      with self.subTest(shape=shape):
        with self.assertRaises(ValueError):
          StubVisionBlock(img_shape=shape)

  def test_constructor_rejects_unknown_numpy_dtype(self) -> None:
    """Checks dtype names are accepted by NumPy before being stored."""

    with self.assertRaises((TypeError, ValueError)):
      StubVisionBlock(img_dtype='not-a-real-numpy-dtype')

  def test_config_request_validates_boolean_state(self) -> None:
    """Checks mutable ConfigRequest flags and their defaults."""

    request = ConfigRequest(requester='consumer',
                            args=(1,),
                            kwargs={'answer': 42},
                            configurator=CameraConfig,
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
                        configurator=CameraConfig,
                        img_source='source',
                        **kwargs)

  def test_image_link_registration_initializes_receive_state(self) -> None:
    """Checks VisionBlock-specific state installed by an ImageLink."""

    source = StubVisionBlock()
    consumer = StubVisionBlock()
    link = ImageLink(source, consumer, name='vision-core-image')

    self.assertEqual(source.img_outputs, [link])
    self.assertEqual(consumer.img_inputs, [link])
    self.assertEqual(tuple(consumer.last_received), (link.name,))
    self.assertIsInstance(consumer.last_received[link.name], ImgData)
    self.assertEqual(consumer.last_received[link.name].id, -1)
    self.assertEqual(consumer.last_received[link.name].img.size, 0)

  def test_begin_and_frequency_counter(self) -> None:
    """Checks handled-image frequency timing and counter reset."""

    block = StubVisionBlock()

    with patch.object(block_module, 'time', return_value=12.5):
      block.begin()
    self.assertEqual(block._last_fps_img, 12.5)

    block.log = Mock()
    block._last_fps_img = 10.0
    block._fps_count = 2
    with patch.object(block_module, 'time', return_value=13.0):
      block._print_freq(img_handled=True)

    block.log.assert_called_once_with(
        logging.INFO, 'Frames handled per second: 1.0')
    self.assertEqual(block._last_fps_img, 13.0)
    self.assertEqual(block._fps_count, 0)

    block.log.reset_mock()
    with patch.object(block_module, 'time', return_value=14.0):
      block._print_freq(img_handled=False)
    block.log.assert_not_called()
    self.assertEqual(block._fps_count, 0)

  def test_finish_closes_inputs_and_unlinks_owned_output(self) -> None:
    """Checks ownership rules when releasing shared-memory handles."""

    block = StubVisionBlock()
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
