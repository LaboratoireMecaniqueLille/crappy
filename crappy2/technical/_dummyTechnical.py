# coding: utf-8
##  @addtogroup technical
# @{

##  @defgroup CmDrive CmDrive
# @{

## @file _dataPicker.py
# @brief Dummy technical
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 13/07/2016

import serial
import time

from crappy2.actuator._dummyActuator import DummyActuator
from crappy2.sensor._dummySensor import DummySensor

from crappy2.technical._meta import motion
from ..sensor import OrientalSensor
from ..actuator import OrientalActuator


class DummyTechnical(motion.Motion):
    """Dummy technical"""

    def __init__(self, baudrate=115200, port='/dev/ttyUSB0', num_device=1):
        super(DummyTechnical, self).__init__(port, baudrate)
        self.baudrate = baudrate
        self.num_device = num_device
        self.port = port
        self.sensor = DummySensor()
        self.actuator = DummyActuator()

    def clear_errors(self):
        print 'clear errors'

    def close(self):
        print 'close'

    def stop(self):
        print 'STOP'

    def reset(self):
        print 'Reset'
