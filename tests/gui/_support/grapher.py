from tests.support.blocks import with_failure_manager
from unittest.mock import patch
from matplotlib import pyplot as mpl_plt
from crappy.blocks.grapher import Grapher
import crappy.blocks.grapher as grapher_module
from tests.support.blocks import BlockTestBase, TestBlock, link

mpl_plt.switch_backend('Agg')


class GrapherTestBase(BlockTestBase):
  def setUp(self) -> None:
    """Tracks Graphers, suppresses notices, and closes stale figures."""

    super().setUp()
    self._warning_patcher = patch.object(grapher_module, 'warn')
    self._warning_patcher.start()
    self._graphers: list[Grapher] = list()
    grapher_module.plt.close('all')

  def tearDown(self) -> None:
    """Closes plotting windows before resetting the Block class state."""

    for grapher in self._graphers:
      grapher.finish()
    grapher_module.plt.close('all')
    self._warning_patcher.stop()

    super().tearDown()

  def _prepare_grapher(self,
                       *labels: tuple[str, str],
                       **kwargs) -> tuple[Grapher, TestBlock]:
    """Creates a linked Grapher and prepares its Matplotlib figure."""

    kwargs.setdefault('plotter', 'mpl')

    source = TestBlock()
    grapher = with_failure_manager(Grapher(*labels, **kwargs))
    link(source, grapher)

    with (patch.object(mpl_plt, 'show'),
          patch.object(mpl_plt, 'pause')):
      grapher.prepare()

    self._graphers.append(grapher)
    return grapher, source

  def _prepare_pyqt_grapher(self,
                            *labels: tuple[str, str],
                            **kwargs) -> tuple[Grapher, TestBlock]:
    """Creates a linked Grapher and prepares its PyQtGraph window."""

    kwargs.setdefault('plotter', 'pyqt')

    source = TestBlock()
    grapher = with_failure_manager(Grapher(*labels, **kwargs))
    link(source, grapher)
    grapher.prepare()

    self._graphers.append(grapher)
    return grapher, source
