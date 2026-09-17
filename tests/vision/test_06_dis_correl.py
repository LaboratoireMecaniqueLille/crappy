# coding: utf-8

from unittest.mock import Mock, patch

import numpy as np

import crappy.blocks.vision.dis_correl as dis_correl_module
from crappy.blocks.vision import DISCorrelProcessor
from crappy.tool.camera_config import Box, DISCorrelConfig, SpotsBoxes

from .vision_test_base import VisionTestBase


class RecordingDISCorrelTool:
  """Small DISCorrelTool double recording setup and processed images."""

  instances: list['RecordingDISCorrelTool'] = list()

  def __init__(self, **kwargs) -> None:
    """Stores constructor arguments and deterministic processing state."""

    self.kwargs = kwargs
    self.box = kwargs['box']
    self.set_box_calls = 0
    self.reference = None
    self.calls = list()
    self.return_value = [10.0, 20.0, 30.0]
    self.offset = (2, -1)
    type(self).instances.append(self)

  def set_box(self) -> None:
    """Records field preparation for the selected box."""

    self.set_box_calls += 1

  def set_img0(self, image: np.ndarray) -> None:
    """Records the copied reference image."""

    self.reference = image

  def get_data(self, image: np.ndarray, residual: bool) -> list[float]:
    """Records the processed image and residual option."""

    self.calls.append((np.copy(image), residual))
    return self.return_value


class TestDISCorrelProcessor(VisionTestBase):
  """Unit tests for the DISCorrelProcessor VisionBlock."""

  def setUp(self) -> None:
    """Resets the processing-tool registry."""

    super().setUp()
    RecordingDISCorrelTool.instances.clear()

  def make_processor(self, **kwargs) -> DISCorrelProcessor:
    """Creates and tracks a processor with a valid default patch."""

    options = {'patch': (1, 2, 3, 4),
               'request_configuration': False}
    options.update(kwargs)
    processor = DISCorrelProcessor(**options)
    self.track_block(processor)
    return processor

  @staticmethod
  def box(y: int = 1, x: int = 2,
          height: int = 3, width: int = 4) -> Box:
    """Builds a configured correlation box."""

    return Box(x_start=x, x_end=x + width,
               y_start=y, y_end=y + height)

  @staticmethod
  def add_image_input(processor: DISCorrelProcessor,
                      name: str = 'dis-image') -> Mock:
    """Registers a minimal input ImageLink double."""

    link = Mock()
    link.name = name
    processor.add_img_input(link)
    return link

  def feed_image(self,
                 processor: DISCorrelProcessor,
                 image: np.ndarray,
                 metadata=None,
                 name: str = 'dis-image') -> None:
    """Installs one received image and makes receive_imgs report it."""

    if name not in processor.last_received:
      self.add_image_input(processor, name)
    if metadata is None:
      metadata = {'ImageUniqueID': 1, 't(s)': 0.1}
    processor.last_received[name].metadata = metadata
    processor.last_received[name].img = image
    processor.receive_imgs = Mock(return_value=[name])

  def test_constructor_sets_default_fields_labels_and_box(self) -> None:
    """Checks default projections, labels, and patch conversion."""

    processor = self.make_processor()

    self.assertEqual(processor._fields, ['x', 'y', 'exx', 'eyy'])
    self.assertEqual(processor.labels, [
      't(s)', 'meta', 'x(pix)', 'y(pix)', 'Exx(%)', 'Eyy(%)', 'overlay',
    ])
    self.assertEqual((processor._patch.x_start, processor._patch.x_end,
                      processor._patch.y_start, processor._patch.y_end),
                     (2, 6, 1, 4))
    self.assertEqual(processor._border, 16)
    self.assertFalse(processor._follow)

  def test_constructor_normalizes_custom_fields_and_residual(self) -> None:
    """Checks scalar/iterable fields and automatic residual labeling."""

    processor = self.make_processor(fields='r',
                                    labels=['time', 'metadata', 'rotation'],
                                    residual=True)
    self.assertEqual(processor._fields, ['r'])
    self.assertEqual(processor.labels,
                     ['time', 'metadata', 'rotation', 'res', 'overlay'])

    field = np.ones((3, 4, 2), dtype=np.float32)
    custom = self.make_processor(fields=field,
                                 labels=['time', 'metadata', 'field'])
    self.assertEqual(len(custom._fields), 1)
    self.assertIs(custom._fields[0], field)

  def test_constructor_validates_patch_fields_and_labels(self) -> None:
    """Checks ROI, projection declarations, and output-label consistency."""

    invalid = (
      {'patch': (0, 0, 2)},
      {'patch': [0, 0, 2, 2]},
      {'patch': (-1, 0, 2, 2)},
      {'patch': (0, 0, 0, 2)},
      {'patch': (0, 0, 2, 0)},
      {'patch': None, 'request_configuration': False},
      {'fields': ['r'], 'labels': None},
      {'fields': []},
      {'fields': ['missing'], 'labels': ['time', 'meta', 'missing']},
      {'fields': ['x', object()],
       'labels': ['time', 'meta', 'x', 'custom']},
      {'labels': ['too', 'few']},
      {'labels': ['same'] * 6},
      {'labels': ['a', 'b', 'c', 'd', 'e', 1]},
    )
    for options in invalid:
      with self.subTest(options=options):
        with self.assertRaises((TypeError, ValueError)):
          self.make_processor(**options)

  def test_constructor_validates_processing_options(self) -> None:
    """Checks DISFlow and residual configuration validation."""

    invalid = (
      {'request_configuration': 1},
      {'alpha': -1},
      {'delta': np.inf},
      {'gamma': 'one'},
      {'finest_scale': -1},
      {'iterations': -1},
      {'gradient_iterations': -1},
      {'init': 1},
      {'patch_size': 0},
      {'patch_stride': 0},
      {'patch_size': 3, 'patch_stride': 3},
      {'residual': 1},
      {'border': 'wide'},
      {'border': (1,)},
      {'border': (1, -1)},
      {'border': -1},
      {'follow': 1},
    )
    for options in invalid:
      with self.subTest(options=options):
        with self.assertRaises((TypeError, ValueError)):
          self.make_processor(**options)

  def test_request_config_reflects_box_and_requirement(self) -> None:
    """Checks optional, required, and disabled source configuration requests."""

    configured = DISCorrelProcessor(patch=(1, 2, 3, 4))
    self.track_block(configured)
    request = configured.request_config('camera')
    self.assertEqual(request.requester, configured.name)
    self.assertEqual(request.img_source, 'camera')
    self.assertIs(request.configurator, DISCorrelConfig)
    self.assertIs(request.kwargs['patch'], configured._patch)
    self.assertFalse(request.required)

    unconfigured = DISCorrelProcessor(patch=None)
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

  def test_prepare_builds_tool_and_sets_projection_box(self) -> None:
    """Checks DISCorrelTool arguments and final field initialization."""

    fields = ['r', 'z']
    processor = self.make_processor(fields=fields,
                                    labels=['time', 'metadata', 'r', 'z'],
                                    alpha=1,
                                    delta=2,
                                    gamma=3,
                                    finest_scale=4,
                                    init=False,
                                    iterations=5,
                                    gradient_iterations=6,
                                    patch_size=9,
                                    patch_stride=7,
                                    residual=True,
                                    border=(9, 10),
                                    follow=True)
    self.add_image_input(processor)
    processor.recv_configs = Mock(return_value={})
    processor._log_queue = Mock()

    with (patch.object(dis_correl_module, 'DISCorrelTool',
                       RecordingDISCorrelTool),
          patch.object(dis_correl_module.VisionBlock,
                       'prepare') as inherited):
      processor.prepare()

    tool = RecordingDISCorrelTool.instances[-1]
    self.assertIs(processor._dis_correl, tool)
    self.assertEqual(tool.kwargs, {
      'box': processor._patch,
      'fields': fields,
      'alpha': 1,
      'delta': 2,
      'gamma': 3,
      'finest_scale': 4,
      'init': False,
      'iterations': 5,
      'gradient_iterations': 6,
      'patch_size': 9,
      'patch_stride': 7,
      'border': (9, 10),
      'follow': True,
    })
    inherited.assert_called_once_with()
    self.assertEqual(tool.set_box_calls, 1)

  def test_prepare_installs_received_box(self) -> None:
    """Checks a source-selected patch replaces constructor state."""

    processor = DISCorrelProcessor(patch=None)
    self.track_block(processor)
    self.add_image_input(processor)
    configured = self.box(2, 3, 4, 5)
    processor.recv_configs = Mock(return_value={'camera': (configured,)})
    processor._log_queue = Mock()

    with (patch.object(dis_correl_module, 'DISCorrelTool',
                       RecordingDISCorrelTool),
          patch.object(dis_correl_module.VisionBlock, 'prepare')):
      processor.prepare()

    self.assertIs(processor._patch, configured)
    self.assertIs(RecordingDISCorrelTool.instances[-1].box, configured)

  def test_prepare_rejects_missing_ambiguous_or_malformed_config(self) -> None:
    """Checks invalid source-configuration result sets."""

    cases = (
      ({}, RuntimeError),
      ({'one': (self.box(),), 'two': (self.box(),)}, NotImplementedError),
      ({'camera': tuple()}, ValueError),
      ({'camera': object()}, TypeError),
    )
    for configs, error in cases:
      with self.subTest(configs=configs, error=error):
        processor = DISCorrelProcessor(patch=None)
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
    """Checks reference copying, residual forwarding, and patch overlay."""

    processor = self.make_processor(residual=True)
    tool = RecordingDISCorrelTool(box=processor._patch)
    processor._dis_correl = tool
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
    processor.last_received['dis-image'].metadata = metadata
    processor.loop()

    sent = processor.send.call_args.args[0]
    self.assertEqual(sent[:-1], [0.2, metadata, 10.0, 20.0, 30.0])
    self.assertIsInstance(sent[-1], SpotsBoxes)
    self.assertIsNot(sent[-1].spot_1, tool.box)
    self.assertEqual((sent[-1].spot_1.x_start,
                      sent[-1].spot_1.x_end,
                      sent[-1].spot_1.y_start,
                      sent[-1].spot_1.y_end),
                     (4, 8, 0, 3))
    np.testing.assert_array_equal(tool.calls[-1][0], image)
    self.assertTrue(tool.calls[-1][1])

  def test_loop_requires_metadata_and_initialized_tool(self) -> None:
    """Checks clear failures for impossible partially initialized states."""

    processor = self.make_processor()
    self.feed_image(processor, np.zeros((3, 4), dtype=np.uint8))
    processor.last_received['dis-image'].metadata = None
    with self.assertRaises(RuntimeError):
      processor.loop()

    processor.last_received['dis-image'].metadata = {
      'ImageUniqueID': 1,
      't(s)': 0.1,
    }
    with self.assertRaises(RuntimeError):
      processor.loop()


if __name__ == '__main__':
  import unittest
  unittest.main()
