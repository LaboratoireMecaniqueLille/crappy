import abc


class Control_Command(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, port, baudrate):
        return

    @abc.abstractmethod
    def new(self, speed):
        pass

    @abc.abstractmethod
    def close(self, position, speed):
        pass
