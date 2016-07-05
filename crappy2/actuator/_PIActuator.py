# coding: utf-8
##  @addtogroup actuator
# @{

##  @defgroup PIActuator PIActuator
# @{

## @file _PIActuator.py
# @brief This class create an axis and opens the corresponding serial port.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 04/07/2016

import numpy as np
import serial
import time
import os
from ._meta import motion
from .._warnings import deprecated as deprecated


# Parameters
# limit = 0.0005 # limit for the eprouvette protection
# offset_=-0.0056
# protection_speed=1000. # nominal speed for the protection
# frequency=500. # refreshing frequency (Hz)
# alpha = 1.05

class PIActuator(motion.MotionActuator):

    def __init__(self, port='/dev/ttyS0', timeout=1, baudrate=9600, ser=None):
        # This class create an axis and opens the corresponding serial port.
        #
        # @param port : str
        #         Path to the corresponding serial port, e.g '/dev/ttyS4'
        # @param baudrate : int, default = 38400
        #         Set the corresponding baud rate.
        # @param timeout : int or float, default = 1
        #         Serial timeout.
        #
        super(PIActuator, self).__init__()
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        if ser is not None:
            self.ser = ser
        else:
            self.ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout)
            a = self.ser.write("%c%cSA%d\r" % (
            1, '0', 10000))  # fixer acceleration de 10 000 a 100 000 microsteps par seconde au carre
            a = self.ser.write("%c%cSV%d\r" % (1, '0', 10000))  # fixer vitesse

    def set_speed(self, speed):
        # TODO
        pass

    def set_position(self, disp):
        # print self.ser
        # """Re-define the speed of the motor. 1 = 0.002 mm/s"""
        # here we should add the physical conversion for the speed
        # self.ser=serial.Serial(self.port_number,self.timeout)
        a = self.ser.write("%c%cMA%d\r" % (1, '0', int(disp)))
        # print a
        # self.ser.close()

    @deprecated(set_position)
    def set_absolute_disp(self, disp):
        ## DEPRECATED: use set_position instead.
        self.set_position(disp)

    @deprecated(None, "Use close function defined in PITechnical instead.")
    def close_port(self):
        ## DEPRECATED: Use close function defined in PITechnical instead.
        # Close the designated port
        self.ser.close()
