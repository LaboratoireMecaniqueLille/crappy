import abc


class MotionActuator(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        return

    @abc.abstractmethod
    def set_speed(self, speed):
        """
        Re-define the speed of the motor.
        """
        pass

    @abc.abstractmethod
    def set_position(self, position, speed):
        """
        Re-define the position of the motor.
        """
        pass
