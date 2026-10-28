"""Importable lifecycle workers for real multiprocessing integration tests."""

from crappy._exceptions import SchedulerStop
from tests.support.blocks import TestBlock


class TestBlockRaisePrepare(TestBlock):
  """Test Block raising an exception from prepare."""

  def prepare(self) -> None:
    """Runs the normal prepare code and then raises."""

    super().prepare()
    raise ValueError


class TestBlockRaiseBegin(TestBlock):
  """Test Block raising an exception from begin."""

  def begin(self) -> None:
    """Runs the normal begin code and then raises."""

    super().begin()
    raise ValueError


class TestBlockRaiseLoop(TestBlock):
  """Test Block raising an exception from loop."""

  def loop(self) -> None:
    """Runs one normal loop iteration and then raises."""

    super().loop()
    raise ValueError


class TestBlockRaiseSchedulerStop(TestBlock):
  """Test Block ending normally with SchedulerStop from loop."""

  def loop(self) -> None:
    """Records the loop call, then requests Scheduler-style shutdown."""

    super().loop()
    raise SchedulerStop


class TestBlockRaiseFinish(TestBlock):
  """Test Block raising an exception from finish."""

  def finish(self) -> None:
    """Runs the normal finish code and then raises."""

    super().finish()
    raise ValueError


class TestBlockT0(TestBlock):
  """Test Block accessing t0 from inside loop."""

  def loop(self) -> None:
    """Runs the regular loop body and then reads t0."""

    super().loop()
    _ = self.t0
