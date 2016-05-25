import abc

class Command(object):

    __metaclass__= abc.ABCMeta

    @abc.abstractmethod
    def __init__(self,device,subdevice,channel,range_num,gain,offset):
        return

    @abc.abstractmethod
    def new(self):
        pass

    @abc.abstractmethod
    def set_cmd(self):
        """
        Send a converted tension value the output.
        """
        pass

    @abc.abstractmethod
    def close(self):
        pass