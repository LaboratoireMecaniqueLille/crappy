# coding: utf-8

"""Optional rendering and GUI dependencies load only for selected features."""

from tests.support.processes import run_python
import os
import multiprocessing as mp
import textwrap
from unittest import TestCase


class TestLazyFeatureImports(TestCase):
  def _run_script(self, script: str) -> None:
    environment = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    result = run_python(['-B', '-c', textwrap.dedent(script)], timeout=45, env=environment)
    if result.returncode == 77:
      self.skipTest('The selected optional backend is unavailable')
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_core_import_and_graph_bookkeeping_need_no_optional_stack(self):
    self._run_script('''
      import importlib.abc
      import sys

      forbidden = {'numpy', 'graphviz', 'tkinter', '_tkinter', 'cv2',
                   'scipy', 'skimage', 'PyQt6', 'matplotlib', 'pyqtgraph'}
      class RejectOptional(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
          if fullname.split('.')[0] in forbidden:
            raise AssertionError('Unused dependency: ' + fullname)
      sys.meta_path.insert(0, RejectOptional())
      import crappy
      from crappy.links import LinkGraph
      graph = LinkGraph()
      graph.add_node('source', int)
      graph.add_node('target', str)
      graph.add_edge('data', 'source', 'target', 'link')
      assert graph.successors('source') == ('target',)
      assert graph.link_names() == ('data',)
      graph.reset()
      assert not graph.nodes
      for name in ('Path', 'Modifier', 'Camera', 'Actuator', 'InOut'):
        assert name in crappy.__all__ and name in dir(crappy)
        assert name not in vars(crappy)
      assert not forbidden.intersection(sys.modules)
    ''')

  def test_numeric_and_numpy_recording_pipelines_need_no_optional_stack(self):
    for method in mp.get_all_start_methods():
      with self.subTest(method=method):
        result = run_python(['-B', '-m', 'tests.unit.framework.lazy_numeric_pipeline', method], timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_graph_rendering_resolves_once_and_preserves_output(self):
    self._run_script('''
      import importlib
      from types import SimpleNamespace
      from unittest.mock import Mock, patch
      module = importlib.import_module('crappy.links.link_graph')
      graph = module.LinkGraph()
      graph.add_node('source', int)
      graph.add_node('target', str)
      graph.add_node('isolated', float)
      graph.add_edge('data', 'source', 'target', 'link')
      graph.add_edge('images', 'source', 'target', 'image')
      drawing = Mock()
      backend = SimpleNamespace(Digraph=Mock(return_value=drawing))
      with patch('crappy._global.import_module', return_value=backend) as load:
        for kwargs, error in (({'links': 1}, TypeError),
                              ({'image_links': None}, TypeError),
                              ({'links': False, 'image_links': False}, ValueError)):
          try:
            graph.display(**kwargs)
          except error:
            pass
          else:
            raise AssertionError('Invalid arguments were accepted')
        load.assert_not_called()
        graph.display()
        load.assert_called_once_with('graphviz')
        backend.Digraph.assert_called_once_with(
            name='Crappy', filename='crappy_graph', format='pdf',
            directory=backend.Digraph.call_args.kwargs['directory'])
        drawing.attr.assert_called_once_with(rankdir='LR')
        drawing.node.assert_any_call('source', label='source\\n(int)')
        drawing.node.assert_any_call('isolated', label='isolated\\n(float)')
        drawing.edge.assert_any_call('source', 'target', label='data', style='solid')
        drawing.edge.assert_any_call('source', 'target', label='images',
                                     style='dashed', penwidth='2')
        drawing.view.assert_called_once_with(cleanup=True)
        drawing.reset_mock()
        graph.display(links=False)
        load.assert_called_once_with('graphviz')
        drawing.edge.assert_called_once_with('source', 'target', label='images',
                                             style='dashed', penwidth='2')
      import crappy
      assert crappy.display_graph.__self__ is module.link_graph
    ''')

  def test_graph_dependency_failures_keep_their_causes_and_are_retriable(self):
    self._run_script('''
      import importlib
      from types import SimpleNamespace
      from unittest.mock import Mock, patch
      from crappy._exceptions import OptionalDependencyError
      module = importlib.import_module('crappy.links.link_graph')
      graph = module.LinkGraph()
      missing = ModuleNotFoundError('missing graphviz', name='graphviz')
      for failure in (missing, RuntimeError('initialization bug'), KeyboardInterrupt()):
        with patch('crappy._global.import_module', side_effect=failure):
          try:
            graph.display()
          except OptionalDependencyError as error:
            assert failure is missing
            assert error.__cause__ is missing
            assert error.import_name == error.install_name == 'graphviz'
          except BaseException as error:
            assert error is failure
          else:
            raise AssertionError('Dependency failure was swallowed')
        assert module.graphviz._module is None
      backend = SimpleNamespace(Digraph=Mock(return_value=Mock()))
      with patch('crappy._global.import_module', return_value=backend):
        graph.display()
      assert module.graphviz._module is backend
    ''')

  def test_actual_graphviz_source_without_launching_a_viewer(self):
    self._run_script('''
      import importlib
      from unittest.mock import patch
      try:
        import graphviz
      except ImportError:
        raise SystemExit(77)
      module = importlib.import_module('crappy.links.link_graph')
      graph = module.LinkGraph()
      graph.add_node('source', int)
      graph.add_node('isolated', str)
      with patch.object(graphviz.Digraph, 'view', autospec=True) as view:
        graph.display()
      drawing = view.call_args.args[0]
      assert 'digraph Crappy' in drawing.source
      assert 'isolated' in drawing.source
      assert 'rankdir=LR' in drawing.source
      assert view.call_args.kwargs == {'cleanup': True}
    ''')

  def test_gui_class_imports_and_construction_do_not_select_a_backend(self):
    self._run_script('''
      import importlib.abc
      import pickle
      import sys
      import warnings

      forbidden = {'tkinter', '_tkinter', 'PyQt6', 'matplotlib', 'pyqtgraph'}
      class RejectGui(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
          if fullname.split('.')[0] in forbidden:
            raise AssertionError('GUI imported before selection: ' + fullname)
      sys.meta_path.insert(0, RejectGui())
      warnings.simplefilter('ignore')
      from crappy.blocks import Button, Canvas, Dashboard, Grapher, StopButton
      import crappy.blocks.dashboard as dashboard
      for backend in ('pyqt', 'tkinter'):
        blocks = [Button(backend=backend), Canvas('missing.png', backend=backend),
                  Dashboard('value', backend=backend), StopButton(backend=backend)]
        assert all(pickle.loads(pickle.dumps(type(block))) is type(block)
                   for block in blocks)
      Grapher(('x', 'y'))
      Grapher(('x', 'y'), plotter='mpl', backend='TkAgg')
      assert 'DashboardWindow' in dir(dashboard)
      assert 'DashboardWindow' not in vars(dashboard)
      assert 'crappy.blocks._dashboard_tkinter' not in sys.modules
      assert not forbidden.intersection(sys.modules)
    ''')

  def test_real_qt_backends_prepare_loop_and_finish_without_tkinter(self):
    self._run_script('''
      import importlib.abc
      from pathlib import Path
      import sys
      import warnings
      from unittest.mock import patch

      class RejectTk(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
          if fullname.split('.')[0] in ('tkinter', '_tkinter'):
            raise AssertionError('Tkinter imported by Qt: ' + fullname)
      sys.meta_path.insert(0, RejectTk())
      try:
        import PyQt6
        import pyqtgraph
      except ImportError:
        raise SystemExit(77)
      import crappy
      from crappy import Block, link
      from crappy.blocks import Button, Canvas, Dashboard, Grapher, StopButton
      from crappy._failure import FailureManager
      warnings.simplefilter('ignore')
      class SourceForLazyGuiTest(Block):
        def loop(self):
          pass
      class SinkForLazyGuiTest(Block):
        def loop(self):
          pass
      source, sink = SourceForLazyGuiTest(), SinkForLazyGuiTest()
      background = Path(crappy.__file__).parent / 'tool' / 'data' / 'pad.png'
      button = Button()
      dashboard = Dashboard('value')
      canvas = Canvas(background)
      stop = StopButton()
      grapher = Grapher(('x', 'y'))
      blocks = [button, dashboard, canvas, stop, grapher]
      link(button, sink)
      for block in (dashboard, canvas, grapher):
        link(source, block)
      for block in blocks:
        block._failure_manager = FailureManager(block.name, 'GUI')
      try:
        for block in blocks:
          block.prepare()
        with patch('crappy._global.import_module',
                   side_effect=AssertionError('First-loop optional import')):
          for block in blocks:
            block.loop()
          for block in blocks:
            block.finish()
        assert 'tkinter' not in sys.modules
        assert '_tkinter' not in sys.modules
        assert 'crappy.blocks._dashboard_tkinter' not in sys.modules
      finally:
        for block in blocks:
          block.finish()
        crappy.reset()
    ''')

  def test_missing_tkinter_fails_before_creating_a_window(self):
    self._run_script('''
      import importlib
      import sys
      import warnings
      from unittest.mock import patch
      from crappy._exceptions import OptionalDependencyError
      warnings.simplefilter('ignore')
      choices = [('button', 'Button', ()), ('canvas', 'Canvas', ('missing.png',)),
                 ('dashboard', 'Dashboard', ('value',)), ('stop_button', 'StopButton', ())]
      cause = ModuleNotFoundError('missing tkinter', name='tkinter')
      for leaf, name, args in choices:
        module = importlib.import_module('crappy.blocks.' + leaf)
        block = getattr(module, name)(*args, backend='tkinter')
        with patch('crappy._global.import_module', side_effect=cause) as load:
          try:
            block._prepare_tkinter()
          except OptionalDependencyError as error:
            assert error.__cause__ is cause
            assert error.import_name == 'tkinter'
          else:
            raise AssertionError('Unavailable Tk backend was accepted')
          load.assert_called_once_with('tkinter')
        assert getattr(block, '_root', None) is None
        assert getattr(block, '_dashboard', None) is None
        assert block._qt_app is None
        assert module.tk._module is None
      assert 'tkinter' not in sys.modules
      assert 'crappy.blocks._dashboard_tkinter' not in sys.modules
    ''')

  def test_selected_tk_backends_resolve_before_their_first_loop(self):
    self._run_script('''
      import importlib
      import sys
      from types import ModuleType, SimpleNamespace
      from unittest.mock import Mock, patch
      import warnings
      from crappy._failure import FailureManager
      warnings.simplefilter('ignore')
      class FakeTclError(Exception):
        pass
      class FakeRoot:
        def __getattr__(self, name):
          return Mock()
      tkinter = ModuleType('tkinter')
      tkinter.Tk = FakeRoot
      tkinter.TclError = FakeTclError
      tkinter.StringVar = Mock()
      tkinter.Label = Mock()
      tkinter.Button = Mock()
      tkinter.TOP, tkinter.BOTH = 'top', 'both'
      figure = SimpleNamespace(Figure=Mock())
      tkagg = SimpleNamespace(FigureCanvasTkAgg=Mock())
      backends = {'tkinter': tkinter, 'matplotlib.figure': figure,
                  'matplotlib.backends.backend_tkagg': tkagg}
      choices = [('button', 'Button', ()), ('canvas', 'Canvas', ('missing.png',)),
                 ('dashboard', 'Dashboard', ('value',)), ('stop_button', 'StopButton', ())]
      with patch.dict(sys.modules, {'tkinter': tkinter}):
        for leaf, name, args in choices:
          module = importlib.import_module('crappy.blocks.' + leaf)
          block = getattr(module, name)(*args, backend='tkinter')
          block._failure_manager = FailureManager(block.name, 'GUI')
          block.recv_last_data = lambda **kwargs: {}
          if name == 'Canvas':
            block._set_tkinter_drawing = Mock()
          with patch('crappy._global.import_module', side_effect=backends.__getitem__) as load:
            block._prepare_tkinter()
            assert module.tk._module is tkinter
            assert ('tkinter',) in [call.args for call in load.call_args_list]
          with patch('crappy._global.import_module',
                     side_effect=AssertionError('First-loop backend import')):
            block.loop()
            block.finish()
          assert block._qt_app is None
    ''')

  def test_dashboard_window_remains_a_real_lazy_public_alias(self):
    self._run_script('''
      import importlib.abc
      import pickle
      import sys
      import crappy.blocks.dashboard as module
      assert 'DashboardWindow' not in vars(module)
      assert 'DashboardWindow' in dir(module)
      assert 'tkinter' not in sys.modules
      class RejectWindow(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
          if fullname == 'crappy.blocks._dashboard_tkinter':
            raise RuntimeError('window import failed')
      guard = RejectWindow()
      sys.meta_path.insert(0, guard)
      try:
        module.DashboardWindow
      except RuntimeError:
        pass
      else:
        raise AssertionError('Failed alias resolution was swallowed')
      finally:
        sys.meta_path.remove(guard)
      assert 'DashboardWindow' not in vars(module)
      try:
        import tkinter
      except ImportError:
        raise SystemExit(77)
      from crappy.blocks.dashboard import DashboardWindow
      from crappy.blocks._dashboard_tkinter import DashboardWindow as implementation
      assert isinstance(DashboardWindow, type)
      assert DashboardWindow is implementation is vars(module)['DashboardWindow']
      assert issubclass(DashboardWindow, tkinter.Tk)
      assert pickle.loads(pickle.dumps(DashboardWindow)) is DashboardWindow
      assert pickle.loads(b'ccrappy.blocks.dashboard\\nDashboardWindow\\n.') is DashboardWindow
      namespace = {}
      exec('from crappy.blocks.dashboard import *', namespace)
      assert namespace['DashboardWindow'] is DashboardWindow
      try:
        module.nonexistent_window
      except AttributeError:
        pass
      else:
        raise AssertionError('Unknown attribute accepted')
    ''')

  def test_matplotlib_agg_handles_errors_without_importing_tkinter(self):
    self._run_script('''
      import importlib.abc
      import sys
      from types import SimpleNamespace
      from unittest.mock import patch
      import warnings
      class RejectTk(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
          if fullname.split('.')[0] in ('tkinter', '_tkinter'):
            raise AssertionError('Tkinter imported by Agg: ' + fullname)
      sys.meta_path.insert(0, RejectTk())
      try:
        import matplotlib
      except ImportError:
        raise SystemExit(77)
      matplotlib.use('Agg')
      from crappy import Block, link
      from crappy.blocks import Grapher
      from crappy._failure import FailureManager
      from crappy.blocks.grapher import _tkinter_errors
      class SourceForLazyAggTest(Block):
        def loop(self):
          pass
      warnings.simplefilter('ignore')
      block = Grapher(('x', 'y'), plotter='mpl', upd_freq=None)
      block._failure_manager = FailureManager(block.name, 'GUI')
      link(SourceForLazyAggTest(), block)
      block.prepare()
      block.recv_all_data_raw = lambda: [{'x': [1], 'y': [2]}]
      try:
        assert _tkinter_errors() == ()
        with patch.dict(sys.modules, {'tkinter': SimpleNamespace()}):
          assert _tkinter_errors() == ()
        block.loop()
        failure = RuntimeError('unrelated draw failure')
        with patch.object(block._canvas, 'draw', side_effect=failure):
          try:
            block.loop()
          except RuntimeError as error:
            assert error is failure
          else:
            raise AssertionError('Unrelated error was swallowed')
        class FakeTclError(Exception):
          pass
        with patch.dict(sys.modules, {'tkinter': SimpleNamespace(TclError=FakeTclError)}):
          assert _tkinter_errors() == (FakeTclError,)
          with patch.object(block._canvas, 'draw', side_effect=FakeTclError):
            block.loop()
          with patch.object(block._canvas, 'draw_idle', side_effect=FakeTclError):
            block._on_press_mpl(SimpleNamespace(key='c'))
        assert 'tkinter' not in sys.modules
      finally:
        block.finish()
    ''')
