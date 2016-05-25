import abc

class MotionSensor(object):

    __metaclass__= abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, port, baudrate):
        return
    
    @abc.abstractmethod
    def get_position(self):
        pass
    