import abc


class Command(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        return

    @abc.abstractmethod
    def new(self):
        pass

    @abc.abstractmethod
    def set_cmd(self, cmd):
        """
        Send a converted tension value the output.
        """
        pass

    @abc.abstractmethod
    def close(self):
        pass
