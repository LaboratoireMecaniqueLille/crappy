# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup MasterBlock MasterBlock
# @{

## @file _meta.py
# @brief Main class for block architecture. All blocks should inherit this class.
#
# @authors Corentin Martel, Robin Siemiatkowski
# @version 0.1
# @date 11/07/2016

from multiprocessing import Process, Pipe
import os
import platform
import ctypes, time
from ..links._link import TimeoutError
import pickle

if platform.system() == "Linux":
    libc = ctypes.CDLL('libc.so.6')


def main_wrapper(b):
    b()


class MasterBlock(object):
    """
    Main class for block architecture. All blocks should inherit this class.

    Methods:
        main()
             Override it to define the main function of this block.
        add_input(Link object)
            Add a Link object as input.
        add_output(Link object)
            Add a Link as output.
        start()
            Start the main() method as a Process.
        stop()
            Stops the process.
    """
    instances = []
    first_call = True

    def __init__(self):
        self.inputs = []
        self.proc = Process(target=main_wrapper, args=(self.loop,))
        self.outputs = []
        self._t0 = time.time()

    def __new__(cls, *args, **kwargs):  # Keeps track of all instances
        instance = super(MasterBlock, cls).__new__(cls, *args, **kwargs)
        instance.instances.append(instance)
        return instance

    @staticmethod
    def close_all_instances():
        if MasterBlock.first_call:
            MasterBlock.first_call = False
            for instance in MasterBlock.instances:
                try:
                    instance.stop()
                except Exception:
                    pass
            for instance in MasterBlock.instances:
                for input_ in instance.inputs:
                    input_.close()
                for output_ in instance.outputs:
                    output_.close()
                MasterBlock.instances.remove(instance)

    def add_output(self, link):
        """
        Add a Link as output.
        Args:
            link: link instance
        """
        self.outputs.append(link)

    def add_input(self, link):
        """
        Add a Link object as input.
        Args:
            link: link instance
        """
        self.inputs.append(link)

    def main(self):
        """
        Override it to define the main function of this block.
        """
        raise NotImplementedError("Must override method main")

    def loop(self):
        try:
            self.main()
        except KeyboardInterrupt:
            self.close_all_instances()
        except Exception as e:
            print e
            self.close_all_instances()

    def start(self):
        """
        Start the main() method as a Process.
        """
        try:
            self.proc.start()
        except KeyboardInterrupt:
            self.proc.terminate()
            self.proc.join()

        except Exception as e:
            print "Exception in MasterBlock: ", e
            if platform.system() == "Linux":
                self.proc.terminate()

            raise  # raise the error to the next level for global shutdown
            # def join(self):
            # self.proc.join()

    def stop(self):
        """
        Stops the process.
        """
        try:
            self.proc.terminate()
            self.proc.join()
        except:
            pass

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, t0):
        self._t0 = t0

    @property
    def pid(self):
      return self.proc.pid


def delay(ms):
    """
    Delay in milliseconds with libc usleep() using ctypes.
    It has a better repeatability than time.sleep()
    """
    ms = int(ms * 1000)
    if platform.system() == "Linux":
        libc.usleep(ms)
    else:
        time.sleep(ms)
