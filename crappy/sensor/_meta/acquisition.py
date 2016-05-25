import abc

class Acquisition:

    __metaclass__= abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass
    
    @abc.abstractmethod
    def new(self):
        pass
    
    @abc.abstractmethod
    def getData(self):
        pass
    