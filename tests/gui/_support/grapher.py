from tests.support.assertions import with_failure_manager
from unittest.mock import patch
from matplotlib import pyplot as mpl_plt
from crappy.blocks.grapher import Grapher
import crappy.blocks.grapher as grapher_module
from tests.support.block_test_case import BlockTestBase
from tests.support.blocks import TestBlock, link

mpl_plt.switch_backend('Agg')


class GrapherTestBase(BlockTestBase):
  def setUp(self) -> None:
    """Tracks Graphers, suppresses notices, and closes stale figures."""

    super().setUp()
    self._warning_patcher = patch.object(grapher_module, 'warn')
    self._warning_patcher.start()
    self.addCleanup(self._warning_patcher.stop)
    self._graphers: list[Grapher] = list()

  def _prepare_grapher(self,
                       *labels: tuple[str, str],
                       **kwargs) -> tuple[Grapher, TestBlock]:
    """Creates a linked Grapher and prepares its Matplotlib figure."""

    kwargs.setdefault('plotter', 'mpl')

    source = TestBlock()
    grapher = with_failure_manager(Grapher(*labels, **kwargs))
    self.addCleanup(grapher.finish)
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
    self.addCleanup(grapher.finish)
    link(source, grapher)
    grapher.prepare()

    self._graphers.append(grapher)
    return grapher, source
