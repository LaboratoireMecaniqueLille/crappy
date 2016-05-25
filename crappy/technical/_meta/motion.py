import abc

class Motion(object):

    __metaclass__= abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, port, baudrate):
        return
    
    @abc.abstractmethod
    def stop(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass
    
    @abc.abstractmethod
    def close(self):
        pass
    
    @abc.abstractmethod
    def clear_errors(self):
        pass