# coding: utf-8
## @addtogroup sensor
# @{

##  @defgroup labjacksensor LabJackSensor
# @{

## @file _labJackSensor.py
# @brief  Sensor class for LabJack devices.
# @author Francois Bari
# @version 0.9
# @date 18/08/2016

from labjack import ljm
from time import time, sleep
from sys import exc_info
from os import getpid
from ._meta import acquisition
from collections import OrderedDict
from multiprocessing import Process, Queue
from Tkinter import Tk, Label
from crappy.technical import LabJack
from warnings import warn


class LabJackSensor(acquisition.Acquisition):
    """Sensor class for LabJack devices."""

    def __init__(self, **kwargs):

        super(LabJackSensor, self).__init__()
        self.Labjack = LabJack(sensor=kwargs)

    def new(self):
        """
        Initialize the device.
        """
        pass

    def start_stream(self):
        self.Labjack.start_stream()

    def get_data(self, mock=None):
        return self.Labjack.get_data()

    def get_stream(self):
        return self.Labjack.get_stream()

    def close(self):
        self.Labjack.close()
