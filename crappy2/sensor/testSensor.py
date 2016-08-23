import numpy

from crappy2.sensor._meta.acquisition import Acquisition


class TestSensor(Acquisition):
    def close(self):
        pass

    def __init__(self):
        super(TestSensor, self).__init__()

    def get_data(self):
        pass

    def new(self):
        pass
