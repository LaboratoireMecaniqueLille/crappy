# coding: utf-8

from multiprocessing import Value
from unittest.mock import Mock, call, patch, sentinel
import logging

import numpy as np

import crappy.blocks.vision.camera as camera_module
from crappy._global import CameraConfigError, PrepareError
from crappy.blocks.camera import DummyCam
from crappy.blocks.vision import CameraSource
from crappy.blocks.vision.block import ConfigRequest
from crappy.camera import Camera as BaseCamera
from crappy.tool.camera_config import CameraConfig

from .vision_test_base import VisionTestBase


class RecordingVisionCamera(BaseCamera):
  """Hardware-free Camera recording lifecycle and acquisition calls."""

  instances: list['RecordingVisionCamera'] = list()

  def __init__(self) -> None:
    """Initializes a trigger setting and deterministic return value."""

    super().__init__()
    self.add_trigger_setting()
    self.open_calls: list[dict] = list()
    self.close_calls = 0
    self.get_image_calls = 0
    self.return_value = None
    type(self).instances.append(self)

  def open(self, **kwargs) -> None:
    """Records options and applies the requested trigger mode."""

    self.open_calls.append(kwargs.copy())
    trigger = kwargs.get('trigger', 'Free run')
    self.set_all(trigger=trigger)

  def get_image(self):
    """Returns the value selected by the current test."""

    self.get_image_calls += 1
    return self.return_value

  def close(self) -> None:
    """Records Camera cleanup."""

    self.close_calls += 1


class RecordingConfig(CameraConfig):
  """Non-GUI CameraConfig double used by CameraSource.configure tests."""

  instances: list['RecordingConfig'] = list()
  shape_value = (4, 5)
  dtype_value = np.dtype('uint16')
  result = ('configured',)
  raise_in: str | None = None

  def __init__(self, camera, log_queue, log_level, max_freq, transform,
               *args, **kwargs) -> None:
    """Records constructor arguments without initializing Tk."""

    self.constructor_args = (camera, log_queue, log_level, max_freq,
                             transform, args, kwargs)
    self.shape = type(self).shape_value
    self.dtype = type(self).dtype_value
    self.start_calls = 0
    self.wait_calls = list()
    self.stop_calls = 0
    type(self).instances.append(self)

  def start(self) -> None:
    """Records startup and optionally raises."""

    self.start_calls += 1
    if type(self).raise_in == 'start':
      raise ValueError('configuration failed')
    if type(self).raise_in == 'keyboard':
      raise KeyboardInterrupt

  def wait_window(self, window) -> None:
    """Records the object passed to Tk's wait helper."""

    self.wait_calls.append(window)

  def stop(self) -> None:
    """Records error cleanup."""

    self.stop_calls += 1

  def get_config(self):
    """Returns deterministic configuration data."""

    return type(self).result


class TestCameraSource(VisionTestBase):
  """Unit tests for the CameraSource VisionBlock."""

  def setUp(self) -> None:
    """Installs deterministic Camera and configurator classes."""

    super().setUp()
    CameraSource.cam_count.clear()
    RecordingVisionCamera.instances.clear()
    RecordingConfig.instances.clear()
    RecordingConfig.shape_value = (4, 5)
    RecordingConfig.dtype_value = np.dtype('uint16')
    RecordingConfig.result = ('configured',)
    RecordingConfig.raise_in = None
    patcher = patch.dict(camera_module.camera_dict,
                         {RecordingVisionCamera.__name__:
                            RecordingVisionCamera})
    patcher.start()
    self.addCleanup(patcher.stop)

  def make_source(self, **kwargs) -> CameraSource:
    """Creates and tracks a physical CameraSource with safe defaults."""

    options = {
      'camera': RecordingVisionCamera.__name__,
      'config': False,
      'img_shape': (2, 3),
      'img_dtype': 'uint8',
    }
    options.update(kwargs)
    source = CameraSource(**options)
    self.track_block(source)
    return source

  @staticmethod
  def add_output(source: CameraSource) -> None:
    """Adds a minimal output marker for source-topology checks."""

    source.img_outputs.append(Mock(name='output-image-link'))

  @staticmethod
  def request(source: CameraSource,
              configurator=RecordingConfig,
              required: bool = True,
              *args,
              **kwargs) -> ConfigRequest:
    """Creates and registers an incoming configuration request."""

    request = ConfigRequest(requester=f'requester-{len(source.config_requests_in)}',
                            args=args,
                            kwargs=kwargs,
                            configurator=configurator,
                            img_source=source.name,
                            connection=Mock(),
                            required=required)
    source.add_config_request_in(request)
    return request

  def test_constructor_sets_camera_options_and_counts_instances(self) -> None:
    """Checks physical/generator naming, forwarded kwargs, and defaults."""

    source = self.make_source(serial='abc')
    self.assertEqual(source._camera_name, RecordingVisionCamera.__name__)
    self.assertEqual(source._camera_kwargs, {'serial': 'abc'})
    self.assertEqual(source.freq, 100)
    self.assertTrue(source._allow_downstream_config)
    self.assertEqual(CameraSource.cam_count[RecordingVisionCamera.__name__], 1)

    generator = lambda _exx, _eyy: np.zeros((2, 3), dtype=np.uint8)
    generated = CameraSource(camera='',
                             config=False,
                             image_generator=generator,
                             img_shape=(2, 3),
                             img_dtype='uint8')
    self.track_block(generated)
    self.assertEqual(generated._camera_name, 'Image Generator')
    self.assertEqual(CameraSource.cam_count['Image Generator'], 1)

  def test_constructor_validates_user_arguments(self) -> None:
    """Checks camera selection and CameraSource-specific options."""

    cases = (
      {'camera': 1},
      {'camera': ''},
      {'camera': 'definitely unknown'},
      {'transform': object()},
      {'config': 1},
      {'allow_downstream_config': 1},
      {'software_trig_label': ''},
      {'software_trig_label': 1},
      {'image_generator': object()},
    )
    for options in cases:
      with self.subTest(options=options):
        defaults = {'camera': RecordingVisionCamera.__name__,
                    'config': False,
                    'img_shape': (2, 3),
                    'img_dtype': 'uint8'}
        defaults.update(options)
        with self.assertRaises((TypeError, ValueError)):
          CameraSource(**defaults)

    deprecated_name = next(iter(camera_module.deprecated_cameras))
    with self.assertRaises(NotImplementedError):
      CameraSource(camera=deprecated_name,
                   config=False,
                   img_shape=(2, 3),
                   img_dtype='uint8')

  def test_prepare_validates_source_image_topology(self) -> None:
    """Checks that CameraSource only accepts outgoing ImageLinks."""

    source = self.make_source()
    with self.assertRaises(IOError):
      source.prepare()

    source.img_inputs.append(Mock())
    self.add_output(source)
    with self.assertRaises(IOError):
      source.prepare()

  def test_prepare_opens_physical_camera_and_switches_trigger(self) -> None:
    """Checks Camera instantiation, open options, and trigger transition."""

    source = self.make_source(trigger='Hdw after config', serial='abc')
    self.add_output(source)

    with patch.object(camera_module.VisionBlock, 'prepare') as inherited:
      source.prepare()

    camera = RecordingVisionCamera.instances[-1]
    self.assertIs(source._camera, camera)
    self.assertEqual(camera.open_calls,
                     [{'trigger': 'Hdw after config', 'serial': 'abc'}])
    self.assertEqual(camera.trigger, 'Hardware')
    inherited.assert_called_once_with()

  def test_prepare_builds_image_generator_camera(self) -> None:
    """Checks synthetic settings, ROI setup, and generated acquisition."""

    generator = Mock(return_value=np.arange(12,
                                            dtype=np.uint8).reshape(3, 4))
    source = CameraSource(camera='',
                          config=False,
                          image_generator=generator,
                          img_shape=(3, 4),
                          img_dtype='uint8')
    self.track_block(source)
    self.add_output(source)

    with patch.object(camera_module.VisionBlock, 'prepare') as inherited:
      source.prepare()

    self.assertIsInstance(source._camera, DummyCam)
    self.assertIn('Exx', source._camera.settings)
    self.assertIn('Eyy', source._camera.settings)
    self.assertIn(source._camera.roi_width_name, source._camera.settings)
    generator.assert_called_once_with(0, 0)
    inherited.assert_called_once_with()

    timestamp, image = source._camera.get_image()
    self.assertIsInstance(timestamp, float)
    np.testing.assert_array_equal(image, generator.return_value)

  def test_prepare_requires_final_image_shape_and_dtype(self) -> None:
    """Checks the non-interactive format requirement."""

    for shape, dtype in ((None, 'uint8'), ((2, 3), None)):
      with self.subTest(shape=shape, dtype=dtype):
        source = self.make_source(img_shape=shape, img_dtype=dtype)
        self.add_output(source)
        with self.assertRaises(ValueError):
          source.prepare()

  def test_prepare_uses_default_configuration_without_requests(self) -> None:
    """Checks fallback to the generic configuration window."""

    source = self.make_source(config=True)
    self.add_output(source)
    self.set_prepare_sync(source)

    with (patch.object(source, 'default_configuration') as default_config,
          patch.object(camera_module.VisionBlock, 'prepare') as inherited):
      source.prepare()

    default_config.assert_called_once_with()
    inherited.assert_called_once_with()

  def test_prepare_runs_and_answers_specialized_configurations(self) -> None:
    """Checks ordered downstream configurators and their responses."""

    source = self.make_source(config=True)
    self.add_output(source)
    self.set_prepare_sync(source)
    first = self.request(source, RecordingConfig, True, 'first', option=1)
    second = self.request(source, RecordingConfig, False, option=2)

    with (patch.object(source, 'configure',
                       side_effect=[('one',), ('two',)]) as configure,
          patch.object(camera_module.VisionBlock, 'prepare') as inherited):
      source.prepare()

    camera = RecordingVisionCamera.instances[-1]
    self.assertEqual(configure.call_args_list, [
      call(camera, RecordingConfig, 'first', option=1),
      call(camera, RecordingConfig, option=2),
    ])
    first.connection.send.assert_called_once_with(('one',))
    second.connection.send.assert_called_once_with(('two',))
    first.connection.close.assert_called_once_with()
    second.connection.close.assert_called_once_with()
    self.assertTrue(first.completed)
    self.assertTrue(second.completed)
    inherited.assert_called_once_with()

  def test_prepare_declines_optional_requests_when_config_is_disabled(self
                                                                      ) -> None:
    """Checks optional requests receive None without interactive config."""

    source = self.make_source(config=False)
    self.add_output(source)
    request = self.request(source, required=False)

    with patch.object(camera_module.VisionBlock, 'prepare') as inherited:
      source.prepare()

    request.connection.send.assert_called_once_with(None)
    request.connection.close.assert_called_once_with()
    self.assertTrue(request.completed)
    inherited.assert_called_once_with()

  def test_prepare_disallowed_downstream_config_uses_default_and_declines(self
                                                                          ) -> None:
    """Checks generic configuration plus optional-request rejection."""

    source = self.make_source(config=True, allow_downstream_config=False)
    self.add_output(source)
    self.set_prepare_sync(source)
    request = self.request(source, required=False)

    with (patch.object(source, 'default_configuration') as default_config,
          patch.object(camera_module.VisionBlock, 'prepare')):
      source.prepare()

    default_config.assert_called_once_with()
    request.connection.send.assert_called_once_with(None)
    self.assertTrue(request.completed)

  def test_prepare_rejects_unserviceable_required_requests_early(self) -> None:
    """Checks required requests fail for both disabling option combinations."""

    for options in ({'config': False},
                    {'config': True, 'allow_downstream_config': False}):
      with self.subTest(options=options):
        source = self.make_source(**options)
        self.add_output(source)
        self.set_prepare_sync(source)
        request = self.request(source, required=True)

        with self.assertRaises(RuntimeError):
          source.prepare()

        request.connection.send.assert_not_called()
        self.assertFalse(request.completed)

  def test_prepare_notices_abort_before_specialized_configuration(self
                                                                  ) -> None:
    """Checks a source stops configuring after another Block fails."""

    source = self.make_source(config=True)
    self.add_output(source)
    self.set_prepare_sync(source)
    source._stop_event.set()
    self.request(source)

    with self.assertRaises(PrepareError):
      source.prepare()

  def test_loop_waits_for_software_trigger(self) -> None:
    """Checks that acquisition is gated by the configured regular label."""

    source = self.make_source(software_trig_label='trigger',
                              display_freq=True)
    source._camera = Mock()
    source.recv_last_data = Mock(return_value={})
    source._print_freq = Mock()

    source.loop()

    source._camera.get_image.assert_not_called()
    source._print_freq.assert_called_once_with(img_handled=False)

  def test_loop_updates_generator_and_builds_standard_metadata(self) -> None:
    """Checks strains, float timestamp conversion, and sent notifications."""

    generator = Mock(return_value=np.arange(12,
                                            dtype=np.uint8).reshape(3, 4))
    source = CameraSource(camera='',
                          config=False,
                          image_generator=generator,
                          software_trig_label='trigger',
                          img_shape=(3, 4),
                          img_dtype='uint8',
                          display_freq=True)
    self.track_block(source)
    self.add_output(source)
    with patch.object(camera_module.VisionBlock, 'prepare'):
      source.prepare()

    source._instance_t0 = Value('d', 100.0)
    source.recv_last_data = Mock(return_value={
      'trigger': True,
      'Exx(%)': 2.5,
      'Eyy(%)': -1.5,
    })
    source.send_img = Mock()
    source.send = Mock()
    source._print_freq = Mock()

    with patch.object(camera_module, 'time', return_value=101.25):
      source.loop()

    generator.assert_has_calls([call(0, 0), call(2.5, -1.5)])
    metadata, image = source.send_img.call_args.args
    self.assertEqual(metadata, {
      't(s)': 1.25,
      'DateTimeOriginal': '1970:01:01 00:01:41',
      'SubsecTimeOriginal': '0.250000',
      'ImageUniqueID': 0,
    })
    np.testing.assert_array_equal(image, generator.return_value)
    source.send.assert_called_once_with({
      't(s)': 1.25,
      'img_index': 0,
      'meta': metadata,
    })
    source._print_freq.assert_called_once_with(img_handled=True)

  def test_loop_preserves_metadata_and_applies_transform(self) -> None:
    """Checks Camera dictionaries, relative time, and image transformation."""

    transform = Mock(side_effect=lambda image: image + 10)
    source = self.make_source(transform=transform)
    source._instance_t0 = Value('d', 100.0)
    source.recv_last_data = Mock(return_value={})
    source.send_img = Mock()
    source.send = Mock()
    source._camera = Mock()
    image = np.arange(4, dtype=np.uint8).reshape(2, 2)
    metadata = {'ImageUniqueID': 7, 't(s)': 101.5, 'camera': 'fake'}
    source._camera.get_image.return_value = (metadata, image)

    source.loop()

    transform.assert_called_once_with(image)
    self.assertEqual(metadata['t(s)'], 1.5)
    sent_metadata, sent_image = source.send_img.call_args.args
    self.assertIs(sent_metadata, metadata)
    np.testing.assert_array_equal(sent_image, image + 10)
    source.send.assert_called_once_with({
      't(s)': 1.5,
      'img_index': 7,
      'meta': metadata,
    })

  def test_loop_handles_no_frame_and_invalid_metadata(self) -> None:
    """Checks idle acquisition and malformed Camera return values."""

    source = self.make_source(display_freq=True)
    source._instance_t0 = Value('d', 100.0)
    source.recv_last_data = Mock(return_value={})
    source.send_img = Mock()
    source._print_freq = Mock()
    source._camera = Mock()
    source._camera.get_image.return_value = None

    source.loop()

    source.send_img.assert_not_called()
    source._print_freq.assert_called_once_with(img_handled=False)

    image = np.zeros((2, 2), dtype=np.uint8)
    for metadata in (1, {}, {'ImageUniqueID': 1}):
      with self.subTest(metadata=metadata):
        source._camera.get_image.return_value = (metadata, image)
        with self.assertRaises(ValueError):
          source.loop()

  def test_finish_closes_only_physical_camera_then_shared_memory(self) -> None:
    """Checks physical and generator Camera cleanup ownership."""

    physical = self.make_source()
    physical._camera = RecordingVisionCamera()
    with patch.object(camera_module.VisionBlock, 'finish') as inherited:
      physical.finish()
    self.assertEqual(physical._camera.close_calls, 1)
    inherited.assert_called_once_with()

    generator = CameraSource(camera='',
                             config=False,
                             image_generator=lambda _x, _y: np.zeros((2, 2)),
                             img_shape=(2, 2),
                             img_dtype='float64')
    self.track_block(generator)
    generator._camera = Mock()
    with patch.object(camera_module.VisionBlock, 'finish') as inherited:
      generator.finish()
    generator._camera.close.assert_not_called()
    inherited.assert_called_once_with()

  def test_configure_runs_window_and_updates_output_format(self) -> None:
    """Checks configurator construction, lifecycle, result, shape, and dtype."""

    transform = Mock()
    source = self.make_source(transform=transform,
                              img_shape=(1, 1),
                              img_dtype='uint8',
                              freq=123)
    source._log_queue = sentinel.log_queue
    source._log_level = logging.WARNING
    camera = RecordingVisionCamera()

    result = source.configure(camera, RecordingConfig,
                              'argument', option=sentinel.option)

    config = RecordingConfig.instances[-1]
    self.assertEqual(config.constructor_args,
                     (camera, sentinel.log_queue, logging.WARNING, 123,
                      transform, ('argument',), {'option': sentinel.option}))
    self.assertEqual(config.start_calls, 1)
    self.assertEqual(config.wait_calls, [config])
    self.assertEqual(config.stop_calls, 0)
    self.assertEqual(result, ('configured',))
    self.assertEqual(source._img_shape, (4, 5))
    self.assertEqual(source._img_dtype, 'uint16')

  def test_configure_validates_camera_and_configurator_types(self) -> None:
    """Checks direct type validation before opening a GUI."""

    source = self.make_source()
    source._log_queue = sentinel.log_queue
    camera = RecordingVisionCamera()

    with self.assertRaises(TypeError):
      source.configure(object(), RecordingConfig)
    with self.assertRaises(TypeError):
      source.configure(camera, object)

  def test_configure_stops_failed_or_interrupted_window(self) -> None:
    """Checks exception translation and KeyboardInterrupt cleanup."""

    source = self.make_source()
    source._log_queue = sentinel.log_queue
    camera = RecordingVisionCamera()

    RecordingConfig.raise_in = 'start'
    with self.assertRaises(CameraConfigError):
      source.configure(camera, RecordingConfig)
    self.assertEqual(RecordingConfig.instances[-1].stop_calls, 1)

    RecordingConfig.raise_in = 'keyboard'
    with self.assertRaises(KeyboardInterrupt):
      source.configure(camera, RecordingConfig)
    self.assertEqual(RecordingConfig.instances[-1].stop_calls, 1)

  def test_configure_reports_missing_log_queue_as_runtime_error(self) -> None:
    """Checks the documented error for unavailable logging infrastructure."""

    source = self.make_source()

    with self.assertRaises(RuntimeError):
      source.configure(RecordingVisionCamera(), RecordingConfig)

  def test_default_configuration_uses_base_configurator(self) -> None:
    """Checks the generic configuration helper."""

    source = self.make_source()
    camera = RecordingVisionCamera()
    source._camera = camera
    source.configure = Mock(return_value=None)

    source.default_configuration()

    source.configure.assert_called_once_with(camera, CameraConfig)


if __name__ == '__main__':
  import unittest
  unittest.main()
