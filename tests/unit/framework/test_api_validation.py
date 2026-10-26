"""Public argument guards and the state they protect."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import warnings

import numpy as np

from crappy import Block, _validation as check
from crappy.blocks import (Camera, DICVE, FakeMachine, Generator,
                           HDFRecorder, Multiplexer, PID,
                           Synchronizer)
from crappy.blocks.vision import (DICVEProcessor, ImageRecorder,
                                  VideoExtensoProcessor)
from crappy.blocks.generator_path import (Constant, Cyclic, CyclicRamp,
                                          Integrator, Ramp, Sine)
from crappy.blocks.generator_path.custom import Custom
from crappy.blocks.generator_path.meta_path import Path as GeneratorPath
from crappy.blocks.schedulers.outputs import FromFile
from crappy.camera.meta_camera.camera_setting import (
  CameraChoiceSetting, CameraScaleSetting)
from crappy.links import ImageLink, Link, LinkGraph, link_graph
from crappy.modifier import (Demux, Diff, DownSampler, Integrate, Mean,
                             Median, MovingAvg, MovingMed, Offset,
                             TrigOnChange, TrigOnValue)
from crappy.tool import ApplyStrainToImage
from crappy.tool.camera_config import Box, SpotsBoxes, Zoom
from tests.support.blocks import BlockTestBase, TestBlock


class TestPublicArgumentGuards(BlockTestBase):
  def setUp(self):
    super().setUp()
    self._warnings = warnings.catch_warnings()
    self._warnings.__enter__()
    self.addCleanup(self._warnings.__exit__, None, None, None)
    warnings.simplefilter('ignore', UserWarning)
    self.addCleanup(setattr, GeneratorPath, 'last_cmd', GeneratorPath.last_cmd)
    GeneratorPath.last_cmd = None

  def test_window_modifiers_require_positive_integer_counts(self):
    for cls in (Mean, Median, MovingAvg, MovingMed, DownSampler):
      for value in (True, 1.5, '2', None, []):
        with self.subTest(cls=cls.__name__, value=value):
          with self.assertRaises(TypeError):
            cls(value)
      for value in (0, -1):
        with self.subTest(cls=cls.__name__, value=value):
          with self.assertRaises(ValueError):
            cls(value)
      self.assertIsNotNone(cls(np.int64(1)))

  def test_label_modifiers_reject_invalid_labels(self):
    constructors = (lambda label: Diff(label), lambda label: Integrate(label),
                    lambda label: TrigOnChange(label),
                    lambda label: TrigOnValue(label, 1))
    for constructor in constructors:
      for value, error in ((1, TypeError), (None, TypeError),
                            ('', ValueError), ('  ', ValueError)):
        with self.subTest(constructor=constructor, value=value):
          with self.assertRaises(error):
            constructor(value)
    with self.assertRaises(TypeError):
      Diff('x', out_label=1)
    with self.assertRaises(ValueError):
      Integrate('x', time_label='')

  def test_offsets_keep_iterables_and_check_numeric_targets(self):
    modifier = Offset(iter(('x', 'y')), iter((1.0, -2.0)))
    self.assertEqual(modifier({'x': 5, 'y': 10}), {'x': 1, 'y': -2})
    for offsets, error in ((('1',), TypeError), ((np.nan,), ValueError),
                            ((np.inf,), ValueError)):
      with self.subTest(offsets=offsets):
        with self.assertRaises(error):
          Offset('x', offsets)
    with self.assertRaises(ValueError):
      Offset(('x', 'x'), (1, 2))

  def test_demux_rejects_bad_shapes_before_mutating_data(self):
    modifier = Demux(('x', 'y'))
    for stream in (np.ones(2), np.ones((2, 1))):
      data = {'stream': stream, 't(s)': np.array([1, 2])}
      with self.subTest(shape=stream.shape):
        with self.assertRaises(ValueError):
          modifier(data)
        self.assertEqual(set(data), {'stream', 't(s)'})
    with self.assertRaises(TypeError):
      Demux('x', mean=1)

  def test_paths_reject_nonfinite_numbers_and_invalid_conditions(self):
    constructors = (lambda value: Constant(None, value),
                    lambda value: Ramp(None, value, init_value=0),
                    lambda value: Sine(None, value, 1),
                    lambda value: Cyclic(None, None, 0, 1, cycles=value),
                    lambda value: Integrator(None, value, 'x', init_value=0))
    for constructor in constructors:
      for value in (np.nan, np.inf, -np.inf):
        with self.subTest(constructor=constructor, value=value):
          with self.assertRaises(ValueError):
            constructor(value)
    for condition, error in ((object(), TypeError), ('x>nan', ValueError),
                              ('x<inf', ValueError), ('delay=-1', ValueError)):
      with self.subTest(condition=condition):
        with self.assertRaises(error):
          Constant(condition, 0)

  def test_signed_paths_and_half_cycles_remain_supported(self):
    self.assertIsNotNone(Ramp(None, -1, init_value=-2))
    self.assertIsNotNone(Integrator(None, -1, 'x', init_value=0))
    self.assertIsNotNone(Cyclic(None, None, -1, 1, cycles=0.5))
    self.assertIsNotNone(CyclicRamp(None, None, -1, 1, cycles=-1,
                                   init_value=0))
    with self.assertRaises(ValueError):
      Cyclic(None, None, 0, 1, cycles=0.25)

  def test_generator_validates_flags_and_path_type_names(self):
    path = [{'type': 'Constant', 'value': 0, 'condition': None}]
    for key in ('repeat', 'spam', 'safe_start'):
      with self.subTest(key=key):
        with self.assertRaises(TypeError):
          Generator(path, **{key: 1})
    with self.assertRaises(ValueError):
      Generator(path, end_delay=np.nan)
    with self.assertRaises(TypeError):
      Generator([{'type': []}])

  def test_pid_keeps_infinite_limits_but_rejects_nan(self):
    pid = PID(-1, out_min=np.inf, out_max=-np.inf,
              i_limit=(np.inf, -np.inf))
    self.assertEqual((pid._out_min, pid._out_max), (-np.inf, np.inf))
    for key in ('kp', 'ki', 'kd', 'out_min', 'out_max'):
      options = {'kp': 1, key: np.nan}
      with self.subTest(key=key):
        with self.assertRaises(ValueError):
          PID(**options)
    with self.assertRaises(ValueError):
      PID(1, i_limit=(1,))

  def test_numeric_blocks_reject_invalid_configuration(self):
    cases = ((Multiplexer, {'interp_freq': np.inf}, ValueError),
             (Synchronizer, {'reference_label': ''}, ValueError),
             (FakeMachine, {'l0': 0}, ValueError),
             (FakeMachine, {'sigma': {'F(N)': -1}}, ValueError),
             (DICVE, {'camera': '', 'image_generator': lambda x, y: None,
                      'patch_size': 8, 'patch_stride': 8}, ValueError),
             (DICVE, {'camera': '', 'image_generator': lambda x, y: None,
                      'alpha': np.nan}, ValueError))
    for cls, options, error in cases:
      with self.subTest(cls=cls.__name__, options=options):
        with self.assertRaises(error):
          cls(**options)

  def test_block_controls_validate_before_starting_a_run(self):
    with patch.object(Block, '_ensure_failure_manager') as manager:
      for method, options in ((Block.start_all, {'allow_root': 1}),
                              (Block.start_all, {'log_level': 'DEBUG'}),
                              (Block.prepare_all, {'log_level': True}),
                              (Block.launch_all, {'no_raise': 1}),
                              (Block.renice_all, {'allow_root': None})):
        with self.subTest(method=method, options=options):
          with self.assertRaises(TypeError):
            method(**options)
      manager.assert_not_called()
    block = TestBlock()
    for value in (np.nan, np.inf):
      with self.assertRaises(ValueError):
        block.freq = value
    with self.assertRaises(TypeError):
      block.debug = 'yes'
    self.assertFalse(block.debug)
    block.log(logging.DEBUG, '')

  def test_link_arguments_fail_before_opening_a_pipe(self):
    upstream, downstream = TestBlock(), TestBlock()
    with patch('crappy.links.link.Pipe') as pipe:
      cases = ({'modifiers': [object()]},
               {'allow_parallel': 'yes'}, {'name': ''})
      for options in cases:
        with self.subTest(options=options):
          with self.assertRaises((TypeError, ValueError)):
            Link(upstream, downstream, **options)
      pipe.assert_not_called()

  def test_image_link_rejects_nonvision_blocks_before_graph_registration(self):
    upstream, downstream = TestBlock(), TestBlock()
    vision = ImageRecorder(save_backend='npy')
    with patch.object(link_graph, 'add_edge') as add_edge:
      for source, target in ((upstream, vision), (vision, downstream)):
        with self.subTest(source=source, target=target):
          with self.assertRaises(NotImplementedError):
            ImageLink(source, target)
      add_edge.assert_not_called()

  def test_unmodified_link_data_is_checked_by_the_parent_once(self):
    upstream, downstream = TestBlock(), TestBlock()
    connection = Link(upstream, downstream)
    with patch.object(check, 'mapping', wraps=check.mapping) as validate:
      upstream.send({'x': 1})
      validate.assert_called_once()
    self.assertEqual(connection.recv(), {'x': 1})

  def test_graph_queries_validate_filters(self):
    graph = LinkGraph()
    graph.add_node('one', TestBlock)
    for method in (graph.ancestors, graph.descendants,
                   graph.predecessors, graph.successors):
      with self.subTest(method=method):
        with self.assertRaises(ValueError):
          method('one', kind='unknown')
        with self.assertRaises(TypeError):
          method([], kind='link')

  def test_camera_settings_reject_bad_types_before_side_effects(self):
    calls = []
    setting = CameraScaleSetting('x', 0, 10, setter=calls.append)
    revision = setting.revision
    for value, error in ((np.nan, ValueError), (np.inf, ValueError),
                          ('1', TypeError)):
      with self.subTest(value=value):
        with self.assertRaises(error):
          setting.value = value
    self.assertEqual(calls, [])
    self.assertEqual(setting.revision, revision)
    with self.assertRaises(ValueError):
      setting.reload(0, np.nan)
    self.assertEqual((setting.lowest, setting.highest), (0, 10))

  def test_choice_settings_validate_assignments_and_reload_atomically(self):
    setting = CameraChoiceSetting('mode', ('a', 'b'))
    for value, error in (('missing', ValueError), (1, TypeError)):
      with self.subTest(value=value):
        with self.assertRaises(error):
          setting.value = value
    with self.assertRaises(TypeError):
      setting.reload(('a', 1))
    self.assertEqual(setting.choices, ('a', 'b'))
    self.assertEqual(setting.value, 'a')

  def test_patch_configuration_fails_before_building_child_collections(self):
    with patch.object(SpotsBoxes, 'set_spots') as set_spots:
      with self.assertRaises(ValueError):
        DICVEProcessor(patches=[(1, 1, 2, 2), (2, 2, 0, 1)])
      set_spots.assert_not_called()
    with self.assertRaises(TypeError):
      SpotsBoxes(spot_1='box')
    for options in ({'blur': 0}, {'blur': 2}, {'border': -1}, {'num_spots': 5}):
      with self.subTest(options=options):
        with self.assertRaises(ValueError):
          VideoExtensoProcessor(**options)

  def test_camera_options_fail_before_creating_workers(self):
    with patch('crappy.blocks.camera.ImageSaver') as saver, \
         patch('crappy.blocks.camera.Displayer') as displayer, \
         patch('crappy.blocks.camera.Manager') as manager:
      cases = ({'save_backend': 'bad'}, {'save_folder': ' '},
               {'img_extension': ' '}, {'save_period': 0},
               {'save_period': -1}, {'save_period': 1.5},
               {'save_period': '2'}, {'save_period': True},
               {'displayer_framerate': np.nan},
               {'displayer_framerate': np.inf},
               {'displayer_framerate': 0},
               {'displayer_framerate': -1},
               {'displayer_backend': 'bad'})
      for options in cases:
        with self.subTest(options=options):
          with self.assertRaises(ValueError):
            Camera('', config=False, image_generator=lambda x, y: None,
                   **options)
      saver.assert_not_called()
      displayer.assert_not_called()
      manager.assert_not_called()

  def test_dis_configuration_is_checked_once_along_the_call_path(self):
    from crappy.blocks.camera_processes import DICVEProcess
    from crappy.tool.image_processing import DICVETool

    with patch.object(check, 'real', wraps=check.real) as real, \
         patch.object(check, 'integer', wraps=check.integer) as integer:
      block = DICVE('', image_generator=lambda x, y: None,
                    method='Pixel precision')
      spots = SpotsBoxes(spot_1=Box(0, 5, 0, 5))
      process = DICVEProcess(patches=spots, method=block._method, alpha=block._alpha,
                             delta=block._delta, gamma=block._gamma,
                             finest_scale=block._finest_scale,
                             iterations=block._iterations,
                             gradient_iterations=block._gradient_iterations,
                             patch_size=block._patch_size,
                             patch_stride=block._patch_stride,
                             border=block._border, safe=block._safe,
                             follow=block._follow, raise_on_exit=block._raise_on_exit)
      DICVETool(spots, method=process._method, alpha=process._alpha,
                delta=process._delta, gamma=process._gamma,
                finest_scale=process._finest_scale,
                iterations=process._iterations,
                gradient_iterations=process._gradient_iterations,
                patch_size=process._patch_size,
                patch_stride=process._patch_stride, border=process._border,
                safe=process._safe, follow=process._follow)
      real_names = ('alpha', 'delta', 'gamma')
      integer_names = ('finest_scale', 'iterations', 'gradient_iterations',
                       'patch_size', 'patch_stride')
      self.assertEqual([call.args[1] for call in real.call_args_list
                        if call.args[1] in real_names], list(real_names))
      self.assertEqual([call.args[1] for call in integer.call_args_list
                        if call.args[1] in integer_names], list(integer_names))

  def test_zoom_and_strain_reject_singular_values(self):
    zoom = Zoom()
    for ratio in (0, -1, np.nan, np.inf):
      with self.subTest(ratio=ratio):
        with self.assertRaises(ValueError):
          zoom.update_zoom(0.5, 0.5, ratio)
    self.assertEqual(zoom, Zoom())
    with self.assertRaises(ValueError):
      ApplyStrainToImage(np.ones((1, 2)))
    strain = ApplyStrainToImage(np.ones((2, 2, 1), dtype=np.uint8))
    with self.assertRaises(ValueError):
      strain(-100, 0)


  def test_file_outputs_reject_empty_or_nonfinite_data(self):
    with TemporaryDirectory() as folder:
      filename = Path(folder) / 'path.csv'
      for contents in ('', '0,0\n1,nan\n'):
        filename.write_text(contents)
        for constructor in (Custom, lambda path: FromFile('x', path)):
          with self.subTest(contents=contents, constructor=constructor):
            with self.assertRaises(ValueError):
              constructor(filename)

  def test_filenames_require_a_name_for_both_strings_and_path_objects(self):
    constructors = (
      ('HDFRecorder', HDFRecorder),
      ('Custom', Custom),
    )
    root = Path(Path.cwd().anchor)
    for name, constructor in constructors:
      for value in ('', ' ', '.', './', str(root), Path(), root, Path(' ')):
        with self.subTest(constructor=name, value=value):
          with self.assertRaises(ValueError):
            constructor(value)
      for value in (1, b'data', object()):
        with self.subTest(constructor=name, value=value):
          with self.assertRaises(TypeError):
            constructor(value)

  def test_input_filenames_reject_directories(self):
    constructors = (Custom,)
    with TemporaryDirectory() as folder:
      for constructor in constructors:
        for value in (folder, Path(folder)):
          with self.subTest(constructor=constructor, value=value):
            with self.assertRaises(IsADirectoryError):
              constructor(value)

  def test_save_folders_reject_files_and_allow_directory_paths(self):
    constructors = (
      lambda path: Camera('', config=False, image_generator=lambda x, y: None,
                           save_folder=path),
      lambda path: ImageRecorder(save_folder=path, save_backend='npy'),
    )
    with TemporaryDirectory() as folder:
      existing = Path(folder) / 'keep.txt'
      existing.write_text('keep')
      missing = Path(folder) / 'new' / 'recording'
      for constructor in constructors:
        for value in (existing, str(existing)):
          with self.subTest(constructor=constructor, value=value):
            with self.assertRaises(NotADirectoryError):
              constructor(value)
        for value in ('', ' ', Path(' ')):
          with self.subTest(constructor=constructor, value=value):
            with self.assertRaises(ValueError):
              constructor(value)
        for value in (None, Path(), '.', Path(Path.cwd().anchor),
                       folder, Path(folder), missing, str(missing)):
          with self.subTest(constructor=constructor, value=value):
            self.assertIsNotNone(constructor(value))
      self.assertEqual(existing.read_text(), 'keep')
      self.assertFalse(missing.exists())
