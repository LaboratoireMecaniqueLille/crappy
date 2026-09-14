# coding: utf-8

from unittest.mock import Mock, patch

import numpy as np

import crappy.blocks.vision.dic_ve as dic_ve_module
from crappy.blocks.vision import DICVEProcessor
from crappy.tool.camera_config import DICVEConfig, SpotsBoxes
from crappy.tool.image_processing import LostPatchError

from .vision_test_base import VisionTestBase


class RecordingDICVETool:
  """Small DICVETool double recording setup and processed images."""

  instances: list['RecordingDICVETool'] = list()

  def __init__(self, **kwargs) -> None:
    """Stores constructor arguments and deterministic processing state."""

    self.kwargs = kwargs
    self.patches = kwargs['patches']
    self.reference = None
    self.images = list()
    self.return_value = ([(1.0, 2.0)], 3.0, 4.0, [(5.0, 6.0)])
    self.raise_on_calculate = False
    type(self).instances.append(self)

  def set_img0(self, image: np.ndarray) -> None:
    """Records the copied reference image."""

    self.reference = image

  def calculate_displacement(self, image: np.ndarray):
    """Records an image and returns data or simulates a lost patch."""

    self.images.append(np.copy(image))
    if self.raise_on_calculate:
      raise LostPatchError('lost patch')
    return self.return_value


class TestDICVEProcessor(VisionTestBase):
  """Unit tests for the DICVEProcessor VisionBlock."""

  def setUp(self) -> None:
    """Resets the processing-tool registry."""

    super().setUp()
    RecordingDICVETool.instances.clear()

  def make_processor(self, **kwargs) -> DICVEProcessor:
    """Creates and tracks a processor with a valid default patch."""

    options = {'patches': [(1, 2, 3, 4)],
               'request_configuration': False}
    options.update(kwargs)
    processor = DICVEProcessor(**options)
    self.track_block(processor)
    return processor

  @staticmethod
  def spots(*patches) -> SpotsBoxes:
    """Builds configured SpotsBoxes for preparation results."""

    spots = SpotsBoxes()
    spots.set_spots(list(patches) or [(1, 2, 3, 4)])
    spots.save_length()
    return spots

  @staticmethod
  def add_image_input(processor: DICVEProcessor,
                      name: str = 'dic-image') -> Mock:
    """Registers a minimal input ImageLink double."""

    link = Mock()
    link.name = name
    processor.add_img_input(link)
    return link

  def feed_image(self,
                 processor: DICVEProcessor,
                 image: np.ndarray,
                 metadata=None,
                 name: str = 'dic-image') -> None:
    """Installs one received image and makes receive_imgs report it."""

    if name not in processor.last_received:
      self.add_image_input(processor, name)
    if metadata is None:
      metadata = {'ImageUniqueID': 1, 't(s)': 0.1}
    processor.last_received[name].metadata = metadata
    processor.last_received[name].img = image
    processor.receive_imgs = Mock(return_value=[name])

  def test_constructor_sets_defaults_labels_and_patch_boxes(self) -> None:
    """Checks default output labels and public patch coordinate conversion."""

    processor = self.make_processor()

    self.assertEqual(processor.labels, [
      't(s)', 'meta', 'Coord(px)', 'Eyy(%)', 'Exx(%)', 'Disp(px)',
      'overlay',
    ])
    patch = processor._patches.spot_1
    self.assertEqual((patch.x_start, patch.x_end,
                      patch.y_start, patch.y_end),
                     (2, 6, 1, 4))

    labels = ('time', 'metadata', 'coordinates', 'eyy', 'exx',
              'displacements')
    custom = self.make_processor(labels=labels)
    self.assertEqual(custom.labels, [*labels, 'overlay'])

  def test_constructor_validates_patches_and_configuration_choice(self
                                                                   ) -> None:
    """Checks patch count, coordinates, dimensions, and required setup."""

    invalid = (
      {'patches': [(0, 0, 2, 2)] * 5},
      {'patches': [(0, 0, 2)]},
      {'patches': [[0, 0, 2, 2]]},
      {'patches': [(-1, 0, 2, 2)]},
      {'patches': [(0, 0, 0, 2)]},
      {'patches': [(0, 0, 2, 0)]},
      {'patches': None, 'request_configuration': False},
      {'patches': [], 'request_configuration': False},
    )
    for options in invalid:
      with self.subTest(options=options):
        with self.assertRaises(ValueError):
          DICVEProcessor(**options)

  def test_constructor_validates_labels_and_processing_options(self) -> None:
    """Checks labels and the arguments forwarded to DICVETool."""

    invalid = (
      {'labels': ['too', 'few']},
      {'labels': ['same'] * 6},
      {'labels': ['a', 'b', 'c', 'd', 'e', 1]},
      {'request_configuration': 1},
      {'method': 'invalid'},
      {'alpha': -1},
      {'delta': np.inf},
      {'gamma': 'one'},
      {'finest_scale': -1},
      {'iterations': -1},
      {'gradient_iterations': -1},
      {'patch_size': 0},
      {'patch_stride': 0},
      {'patch_size': 3, 'patch_stride': 3},
      {'border': -0.1},
      {'border': 1},
      {'safe': 1},
      {'follow': 1},
      {'raise_on_patch_exit': 1},
    )
    for options in invalid:
      with self.subTest(options=options):
        with self.assertRaises((TypeError, ValueError)):
          self.make_processor(**options)

  def test_request_config_reflects_patches_and_requirement(self) -> None:
    """Checks optional, required, and disabled source configuration requests."""

    configured = DICVEProcessor(patches=[(1, 2, 3, 4)])
    self.track_block(configured)
    request = configured.request_config('camera')
    self.assertEqual(request.requester, configured.name)
    self.assertEqual(request.img_source, 'camera')
    self.assertIs(request.configurator, DICVEConfig)
    self.assertIs(request.kwargs['patches'], configured._patches)
    self.assertFalse(request.required)

    unconfigured = DICVEProcessor(patches=None)
    self.track_block(unconfigured)
    self.assertTrue(unconfigured.request_config('camera').required)

    disabled = self.make_processor(request_configuration=False)
    self.assertIsNone(disabled.request_config('camera'))

  def test_prepare_validates_supported_topology(self) -> None:
    """Checks the one-image-input, no-other-input topology."""

    processor = self.make_processor()
    with self.assertRaises(IOError):
      processor.prepare()

    self.add_image_input(processor)
    processor.img_outputs.append(Mock())
    with self.assertRaises(IOError):
      processor.prepare()

    processor.img_outputs.clear()
    processor.inputs.append(Mock())
    with self.assertRaises(IOError):
      processor.prepare()

    processor.inputs.clear()
    self.add_image_input(processor, 'second-image')
    with self.assertRaises(IOError):
      processor.prepare()

  def test_prepare_builds_tool_with_user_patches(self) -> None:
    """Checks DICVETool arguments and inherited image-buffer preparation."""

    processor = self.make_processor(method='Parabola',
                                    alpha=1,
                                    delta=2,
                                    gamma=3,
                                    finest_scale=4,
                                    iterations=5,
                                    gradient_iterations=6,
                                    patch_size=9,
                                    patch_stride=7,
                                    border=0.3,
                                    safe=False,
                                    follow=False)
    self.add_image_input(processor)
    processor.recv_configs = Mock(return_value={})
    processor._log_queue = Mock()

    with (patch.object(dic_ve_module, 'DICVETool', RecordingDICVETool),
          patch.object(dic_ve_module.VisionBlock, 'prepare') as inherited):
      processor.prepare()

    tool = RecordingDICVETool.instances[-1]
    self.assertIs(processor._disve, tool)
    self.assertEqual(tool.kwargs, {
      'patches': processor._patches,
      'method': 'Parabola',
      'alpha': 1,
      'delta': 2,
      'gamma': 3,
      'finest_scale': 4,
      'iterations': 5,
      'gradient_iterations': 6,
      'patch_size': 9,
      'patch_stride': 7,
      'border': 0.3,
      'safe': False,
      'follow': False,
    })
    inherited.assert_called_once_with()

  def test_prepare_installs_received_patches(self) -> None:
    """Checks source-selected patches replace constructor state."""

    processor = DICVEProcessor(patches=None)
    self.track_block(processor)
    self.add_image_input(processor)
    configured = self.spots((2, 3, 4, 5))
    processor.recv_configs = Mock(return_value={'camera': (configured,)})
    processor._log_queue = Mock()

    with (patch.object(dic_ve_module, 'DICVETool', RecordingDICVETool),
          patch.object(dic_ve_module.VisionBlock, 'prepare')):
      processor.prepare()

    self.assertIs(processor._patches, configured)
    self.assertIs(RecordingDICVETool.instances[-1].patches, configured)

  def test_prepare_rejects_missing_ambiguous_or_malformed_config(self) -> None:
    """Checks invalid source-configuration result sets."""

    cases = (
      ({}, RuntimeError),
      ({'one': (self.spots(),), 'two': (self.spots(),)},
       NotImplementedError),
      ({'camera': tuple()}, ValueError),
      ({'camera': object()}, TypeError),
    )
    for configs, error in cases:
      with self.subTest(configs=configs, error=error):
        processor = DICVEProcessor(patches=None)
        self.track_block(processor)
        self.add_image_input(processor)
        processor.recv_configs = Mock(return_value=configs)
        processor._log_queue = Mock()
        with self.assertRaises(error):
          processor.prepare()

  def test_loop_skips_when_no_new_image(self) -> None:
    """Checks idle polling and handled-image frequency accounting."""

    processor = self.make_processor(display_freq=True)
    processor.receive_imgs = Mock(return_value=[])
    processor._print_freq = Mock()

    processor.loop()

    processor._print_freq.assert_called_once_with(img_handled=False)

  def test_loop_sets_reference_then_sends_results_and_overlay(self) -> None:
    """Checks reference copying and formatted result publication."""

    processor = self.make_processor()
    tool = RecordingDICVETool(patches=processor._patches)
    processor._disve = tool
    processor.send = Mock()
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    metadata = {'ImageUniqueID': 1, 't(s)': 0.1}
    self.feed_image(processor, image, metadata)

    processor.loop()

    self.assertTrue(processor._img0_set)
    self.assertIsNot(tool.reference, image)
    np.testing.assert_array_equal(tool.reference, image)
    processor.send.assert_not_called()

    image[:] = 20
    metadata = {'ImageUniqueID': 2, 't(s)': 0.2}
    processor.last_received['dic-image'].metadata = metadata
    processor.loop()

    processor.send.assert_called_once_with([
      0.2, metadata, [(1.0, 2.0)], 3.0, 4.0, [(5.0, 6.0)],
      tool.patches,
    ])
    np.testing.assert_array_equal(tool.images[-1], image)
    self.assertEqual(processor._last_data, tool.return_value)

  def test_loop_handles_lost_patch_according_to_option(self) -> None:
    """Checks fatal loss and tolerated loss with final overlay clearing."""

    image = np.zeros((3, 4), dtype=np.uint8)
    metadata = {'ImageUniqueID': 3, 't(s)': 0.3}

    fatal = self.make_processor(raise_on_patch_exit=True)
    fatal._img0_set = True
    fatal._disve = RecordingDICVETool(patches=fatal._patches)
    fatal._disve.raise_on_calculate = True
    self.feed_image(fatal, image, metadata, name='fatal-image')
    with self.assertRaises(LostPatchError):
      fatal.loop()
    self.assertTrue(fatal._lost_patch)

    tolerated = self.make_processor(raise_on_patch_exit=False)
    tolerated._img0_set = True
    tolerated._disve = RecordingDICVETool(patches=tolerated._patches)
    tolerated._last_data = ([(1, 2)], 3, 4, [(5, 6)])
    tolerated._disve.raise_on_calculate = True
    tolerated.send = Mock()
    self.feed_image(tolerated, image, metadata, name='tolerated-image')

    tolerated.loop()

    self.assertTrue(tolerated._lost_patch)
    tolerated.send.assert_called_once_with([
      0.3, metadata, [(1, 2)], 3, 4, [(5, 6)], list(),
    ])
    tolerated.receive_imgs.reset_mock()
    tolerated.loop()
    tolerated.receive_imgs.assert_not_called()

  def test_loop_requires_metadata_and_initialized_tool(self) -> None:
    """Checks clear failures for impossible partially initialized states."""

    processor = self.make_processor()
    self.feed_image(processor, np.zeros((3, 4), dtype=np.uint8),
                    metadata={})
    processor.last_received['dic-image'].metadata = None
    with self.assertRaises(RuntimeError):
      processor.loop()

    processor.last_received['dic-image'].metadata = {
      'ImageUniqueID': 1,
      't(s)': 0.1,
    }
    with self.assertRaises(RuntimeError):
      processor.loop()


if __name__ == '__main__':
  import unittest
  unittest.main()
