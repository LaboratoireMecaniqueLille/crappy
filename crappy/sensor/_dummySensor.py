# coding: utf-8
import time

from ._meta import acquisition


class DummySensor(acquisition.Acquisition):
    """Mock a sensor and return the time. Use it for testing."""

    def __init__(self, *args, **kwargs):
        super(DummySensor, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def get_data(self):
        """Return time."""
        t = time.time()
        return t

    def new(self, *args):
        """Do nothing."""
        pass

    def close(self, *args):
        """Do nothing."""
        pass
