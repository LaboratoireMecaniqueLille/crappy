# coding: utf-8

"""Headless contracts shared by the native Qt and legacy Tk GUI Blocks."""

from multiprocessing import Value
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel
import locale
import os
import subprocess
import sys
import textwrap

import crappy.blocks.button as button_module
import crappy.blocks.canvas as canvas_module
import crappy.blocks.dashboard as dashboard_module
import crappy.blocks.stop_button as stop_button_module
from crappy._global import OptionalModule

from ..block import BlockTestBase, TestBlock, link


GUI_BLOCKS = (
  (button_module.Button, button_module, ()),
  (canvas_module.Canvas, canvas_module, ('background.png',)),
  (dashboard_module.Dashboard, dashboard_module, (('a', 'b'),)),
  (stop_button_module.StopButton, stop_button_module, ()),
)


class GUIBlockTestBase(BlockTestBase):
  """Suppress migration notices while retaining the normal Block harness."""

  def setUp(self) -> None:
    super().setUp()
    for _, module, _ in GUI_BLOCKS:
      patcher = patch.object(module, 'warn')
      patcher.start()
      self.addCleanup(patcher.stop)

  @staticmethod
  def _connect(block) -> None:
    """Give each GUI Block just the Links its prepare method requires."""

    if isinstance(block, button_module.Button):
      link(block, TestBlock())
    elif not isinstance(block, stop_button_module.StopButton):
      link(TestBlock(), block)


class TestGUIBackends(GUIBlockTestBase):
  """Exercise backend selection and lifecycle without creating any windows."""

  def test_default_and_explicit_backends_do_not_create_gui_objects(
      self) -> None:
    """Construction stays headless and PyQt is the default for all four."""

    for block_type, module, args in GUI_BLOCKS:
      for backend in (None, 'pyqt', 'tkinter'):
        with self.subTest(block=block_type.__name__, backend=backend):
          with (patch.object(module, 'QtCore', sentinel.unused_qt_core),
                patch.object(module, 'QtWidgets', sentinel.unused_qt_widgets),
                patch.object(block_type, '_prepare_pyqt') as pyqt,
                patch.object(block_type, '_prepare_tkinter') as tkinter):
            kwargs = {} if backend is None else {'backend': backend}
            block = block_type(*args, **kwargs)
            selected = (block.backend
                        if isinstance(block, canvas_module.Canvas)
                        else block._backend)
            self.assertEqual(selected, backend or 'pyqt')
            self.assertIsNone(block._qt_app)
            self.assertIsNone(block._qt_window)
            pyqt.assert_not_called()
            tkinter.assert_not_called()

  def test_invalid_backends_are_rejected(self) -> None:
    """Legacy Matplotlib backend names and wrong types fail at construction."""

    for block_type, _, args in GUI_BLOCKS:
      for backend in ('', ' ', 'invalid', 'Agg', 'TkAgg', 'PyQt6', None, 1):
        with self.subTest(block=block_type.__name__, backend=backend):
          error = ValueError if isinstance(backend, str) else TypeError
          with self.assertRaises(error):
            block_type(*args, backend=backend)

  def test_prepare_dispatches_only_to_the_selected_backend(self) -> None:
    """Tk preparation never initializes Qt, and vice versa."""

    for block_type, _, args in GUI_BLOCKS:
      for backend in ('pyqt', 'tkinter'):
        with self.subTest(block=block_type.__name__, backend=backend):
          block = block_type(*args, backend=backend)
          self._connect(block)
          with (patch.object(block, '_prepare_pyqt') as pyqt,
                patch.object(block, '_prepare_tkinter') as tkinter):
            block.prepare()
            selected, unused = ((pyqt, tkinter) if backend == 'pyqt'
                                else (tkinter, pyqt))
            selected.assert_called_once_with()
            unused.assert_not_called()

  def test_prepare_rejects_missing_and_forbidden_links_before_gui_setup(
      self) -> None:
    """Topology errors must not leave a partially opened GUI behind."""

    for block_type, _, args in GUI_BLOCKS:
      for backend in ('pyqt', 'tkinter'):
        for invalid in ('missing', 'forbidden'):
          if (block_type is stop_button_module.StopButton and
              invalid == 'missing'):
            continue
          with self.subTest(block=block_type.__name__, backend=backend,
                            invalid=invalid):
            block = block_type(*args, backend=backend)
            if invalid == 'forbidden':
              self._connect(block)
              if block_type in (button_module.Button,
                                stop_button_module.StopButton):
                link(TestBlock(), block)
              else:
                link(block, TestBlock())
            with (patch.object(block, '_prepare_pyqt') as pyqt,
                  patch.object(block, '_prepare_tkinter') as tkinter):
              with self.assertRaises(IOError):
                block.prepare()
              pyqt.assert_not_called()
              tkinter.assert_not_called()

    button = stop_button_module.StopButton()
    link(button, TestBlock())
    with self.assertRaises(IOError):
      button.prepare()

  def test_finish_is_safe_before_and_after_partial_preparation(self) -> None:
    """Repeated cleanup closes only this Block's window, not the Qt app."""

    for block_type, module, args in GUI_BLOCKS:
      with self.subTest(block=block_type.__name__):
        block = block_type(*args)
        block.finish()
        block.finish()
        window, application = Mock(), Mock()
        block._qt_window = window
        block._qt_app = application
        tk_window = Mock()
        tk_window.destroy.side_effect = module.tk.TclError
        if isinstance(block, dashboard_module.Dashboard):
          block._dashboard = tk_window
        else:
          block._root = tk_window
        block.finish()
        block.finish()
        self.assertEqual(window.close.call_count, 2)
        self.assertEqual(application.processEvents.call_count, 2)
        application.quit.assert_not_called()

  def test_existing_widget_application_is_reused(self) -> None:
    """Reuse an existing widget app without changing its environment."""

    class Application:
      pass

    existing = Application()
    original = os.path.join('camera', 'cv2', 'qt', 'plugins')
    for block_type, module, _ in GUI_BLOCKS:
      with self.subTest(block=block_type.__name__):
        with (patch.dict(os.environ,
                         {'QT_QPA_PLATFORM_PLUGIN_PATH': original}),
              patch.object(module, 'QtCore') as core,
              patch.object(module, 'QtWidgets',
                           SimpleNamespace(QApplication=Application))):
          core.QCoreApplication.instance.return_value = existing
          self.assertIs(block_type._get_application(), existing)
          self.assertEqual(os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'], original)

  def test_non_widget_application_is_rejected(self) -> None:
    """A QCoreApplication cannot be reused for GUI widgets."""

    for block_type, module, _ in GUI_BLOCKS:
      with self.subTest(block=block_type.__name__):
        with (patch.object(module, 'QtCore') as core,
              patch.object(module, 'QtWidgets',
                           SimpleNamespace(QApplication=type('App', (), {})))):
          core.QCoreApplication.instance.return_value = sentinel.core_app
          with self.assertRaisesRegex(RuntimeError,
                                      'non-widget Qt application'):
            block_type._get_application()

  def test_application_creation_restores_plugin_path_and_numeric_locale(
      self) -> None:
    """OpenCV plugin filtering and locale restoration also survive failures."""

    variable = 'QT_QPA_PLATFORM_PLUGIN_PATH'
    opencv = os.path.join('camera', 'cv2', 'qt', 'plugins')
    unrelated = os.path.join('application', 'plugins')
    cases = ((None, None), (opencv, None), (unrelated, unrelated),
             (os.pathsep.join((opencv, unrelated, opencv)), unrelated))
    for block_type, module, _ in GUI_BLOCKS:
      for original, expected in cases:
        for fails in (False, True):
          with self.subTest(block=block_type.__name__, path=original,
                            fails=fails):
            def create_application(args):
              self.assertEqual(args, [])
              self.assertEqual(os.environ.get(variable), expected)
              if fails:
                raise RuntimeError('application failed')
              return sentinel.application

            with (patch.dict(os.environ),
                  patch.object(module, 'QtCore') as core,
                  patch.object(module, 'QtWidgets') as widgets,
                  patch.object(module.locale, 'setlocale',
                               return_value='original') as setlocale):
              if original is None:
                os.environ.pop(variable, None)
              else:
                os.environ[variable] = original
              core.QCoreApplication.instance.return_value = None
              widgets.QApplication.side_effect = create_application
              if fails:
                with self.assertRaisesRegex(RuntimeError,
                                            'application failed'):
                  block_type._get_application()
              else:
                self.assertIs(block_type._get_application(),
                              sentinel.application)
              self.assertEqual(os.environ.get(variable), original)
              self.assertEqual(setlocale.call_args_list[0].args,
                               (locale.LC_NUMERIC,))
              self.assertEqual(setlocale.call_args_list[-1].args,
                               (locale.LC_NUMERIC, 'original'))

  def test_qt_callback_errors_are_retained_and_raised_by_loop(self) -> None:
    """Qt signal handlers must not let exceptions abort the interpreter."""

    for block_type, callback, handler in (
        (button_module.Button, '_next_step', '_next_step_pyqt'),
        (stop_button_module.StopButton, '_clicked', '_clicked_pyqt')):
      for error in (RuntimeError('callback failed'), KeyboardInterrupt()):
        with self.subTest(block=block_type.__name__,
                          error=type(error).__name__):
          block = block_type()
          block._qt_app = Mock()
          block._qt_button = Mock()
          with patch.object(block, callback, side_effect=error):
            getattr(block, handler)(False)
          self.assertIs(block._callback_error, error)
          block._qt_button.setEnabled.assert_called_once_with(False)
          with self.assertRaises(type(error)) as raised:
            block.loop()
          self.assertIs(raised.exception, error)
          block._qt_app.processEvents.assert_called_once_with()

  def test_button_counter_and_sending_do_not_depend_on_tk_variables(
      self) -> None:
    """The counter is a plain integer before either GUI is prepared."""

    for backend in ('pyqt', 'tkinter'):
      with self.subTest(backend=backend):
        button = button_module.Button(backend=backend, spam=True)
        button._instance_t0 = Value('d', 10.0)
        sink = TestBlock()
        link(button, sink)
        self.assertEqual(button._step, 0)
        with patch.object(button_module, 'time', return_value=12.0):
          button.begin()
        self.assertEqual(sink.inputs[0].recv(), {'t(s)': 2.0, 'step': 0})

  def test_optional_gui_dependencies_are_reported_only_on_use(self) -> None:
    """A fresh process can import/construct all Blocks without Qt or MPL."""

    script = textwrap.dedent('''
        import importlib.abc
        import sys
        import warnings
        from unittest.mock import patch

        class NoGuiDependencies(importlib.abc.MetaPathFinder):
          def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in ('PyQt6', 'matplotlib', 'pyqtgraph'):
              raise ModuleNotFoundError(fullname + ' unavailable')

        sys.meta_path.insert(0, NoGuiDependencies())
        warnings.simplefilter('ignore', UserWarning)
        from crappy import Block, link
        from crappy.blocks import Button, Canvas, Dashboard, StopButton

        class Peer(Block):
          def loop(self):
            pass

        for cls, args in ((Button, ()), (Canvas, ('background.png',)),
                          (Dashboard, ('a',)), (StopButton, ())):
          tkinter = cls(*args, backend='tkinter')
          with patch.object(tkinter, '_prepare_tkinter') as prepare:
            if cls is Button:
              link(tkinter, Peer())
            elif cls is not StopButton:
              link(Peer(), tkinter)
            tkinter.prepare()
            prepare.assert_called_once_with()
          qt = cls(*args)
          if cls is Button:
            link(qt, Peer())
          elif cls is not StopButton:
            link(Peer(), qt)
          try:
            qt.prepare()
          except RuntimeError as error:
            assert 'PyQt6' in str(error), error
          else:
            raise AssertionError('Missing PyQt6 was not reported')
          qt.finish()
        assert not any(name.startswith(('PyQt6', 'matplotlib', 'pyqtgraph'))
                       for name in sys.modules)
    ''')
    result = subprocess.run([sys.executable, '-c', script],
                            capture_output=True, text=True, timeout=30)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_missing_qt_placeholder_has_a_clear_error(self) -> None:
    """Missing Qt reports the optional package, not an AttributeError."""

    for block_type, module, _ in GUI_BLOCKS:
      with self.subTest(block=block_type.__name__):
        with patch.object(module, 'QtCore', OptionalModule('PyQt6.QtCore')):
          with self.assertRaisesRegex(RuntimeError, 'Missing module: PyQt6'):
            block_type._get_application()


class TestGUIArguments(GUIBlockTestBase):
  """Validate the stricter public arguments without graphical dependencies."""

  def test_button_labels_and_boolean_flags(self) -> None:
    """Reject ambiguous labels and non-boolean sending options."""

    for name in ('label', 'time_label'):
      for value in ('', ' ', None, 1):
        with self.subTest(argument=name, value=value):
          error = ValueError if isinstance(value, str) else TypeError
          with self.assertRaises(error):
            button_module.Button(**{name: value})
    with self.assertRaises(ValueError):
      button_module.Button(label='same', time_label='same')
    for name in ('spam', 'send_0'):
      for value in (0, 1, None, 'True'):
        with self.subTest(argument=name, value=value):
          with self.assertRaises(TypeError):
            button_module.Button(**{name: value})

  def test_dashboard_labels_and_precision(self) -> None:
    """Require non-empty labels and a non-negative integer precision."""

    for labels in ('', ' ', [], [''], [' '], ['a', '']):
      with self.subTest(labels=labels):
        with self.assertRaises(ValueError):
          dashboard_module.Dashboard(labels)
    for labels in (None, 1, b'a', [1], ['a', None]):
      with self.subTest(labels=labels):
        with self.assertRaises(TypeError):
          dashboard_module.Dashboard(labels)
    for digits, error in ((-1, ValueError), (True, TypeError),
                          (1.5, TypeError), ('2', TypeError)):
      with self.subTest(digits=digits):
        with self.assertRaises(error):
          dashboard_module.Dashboard('a', nb_digits=digits)
    dashboard = dashboard_module.Dashboard(iter(('b', 'a')), nb_digits=0)
    self.assertEqual(dashboard._dash_labels, ['b', 'a'])
    self.assertEqual(dashboard._nb_digits, 0)

  def test_canvas_window_image_and_color_arguments(self) -> None:
    """Validate image paths, sizes, titles, and finite color bounds early."""

    cases = (
      ('image_path', '', ValueError), ('image_path', ' ', ValueError),
      ('image_path', None, TypeError),
      ('title', '', ValueError), ('title', None, TypeError),
      ('window_size', (0, 2), ValueError),
      ('window_size', (1, float('inf')), ValueError),
      ('window_size', (float('nan'), 1), ValueError),
      ('window_size', [1, 2], TypeError),
      ('color_range', (1, 1), ValueError),
      ('color_range', (float('nan'), 1), ValueError),
      ('color_range', (1, float('inf')), ValueError),
      ('color_range', [1, 2], TypeError),
    )
    for name, value, error in cases:
      with self.subTest(argument=name, value=value):
        kwargs = {'image_path': 'background.png', name: value}
        with self.assertRaises(error):
          canvas_module.Canvas(**kwargs)
    canvas = canvas_module.Canvas('background.png', color_range=(5, 1),
                                 window_size=(3.5, 2.25))
    self.assertEqual(canvas.color_range, (1, 5))
    self.assertEqual(canvas._window_size, (3.5, 2.25))

  def test_canvas_draw_elements_are_validated_and_copied(self) -> None:
    """Overlay validation and copying are independent of the selected GUI."""

    valid = {'type': 'text', 'coord': (1, 2), 'label': 'a', 'text': 'A = %.1f'}
    invalid = (({}, ValueError), ('text', TypeError),
               ({**valid, 'type': 'unknown'}, ValueError),
               ({**valid, 'type': 1}, TypeError),
               ({**valid, 'coord': (float('nan'), 2)}, ValueError),
               ({**valid, 'coord': [1, 2]}, TypeError),
               ({**valid, 'label': ''}, ValueError),
               ({**valid, 'label': None}, TypeError),
               ({**valid, 'text': 1}, TypeError),
               ({'type': 'dot_text', 'coord': (1, 2)}, ValueError))
    for element, error in invalid:
      with self.subTest(element=element):
        with self.assertRaises(error):
          canvas_module.Canvas('background.png', draw=[element])
    for draw in ('text', b'text', iter([valid])):
      with self.subTest(draw=draw):
        with self.assertRaises(TypeError):
          canvas_module.Canvas('background.png', draw=draw)
    canvas = canvas_module.Canvas('background.png', draw=(valid,
                                  {'type': 'time', 'coord': (3, 4)}))
    valid['text'] = 'changed'
    self.assertEqual(canvas._draw[0]['text'], 'A = %.1f')
    self.assertEqual(canvas._draw[1], {'type': 'time', 'coord': (3, 4)})
