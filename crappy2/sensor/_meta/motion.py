import abc


class MotionSensor(object):

    __metaclass__= abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, port, baudrate):
        self.baudrate = baudrate
        self.port = port
        return
    
    @abc.abstractmethod
    def get_position(self):
        pass