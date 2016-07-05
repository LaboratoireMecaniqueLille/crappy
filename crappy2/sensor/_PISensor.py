# coding: utf-8
##  @addtogroup sensor
# @{

##  @defgroup PISensor PISensor
# @{

## @file _PISensor.py
# @brief  This class create an axis and opens the corresponding serial port.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 29/06/2016

import numpy as np
import serial
import time
import os
from ._meta import motion


# Parameters
# limit = 0.0005 # limit for the eprouvette protection
# offset_=-0.0056
# protection_speed=1000. # nominal speed for the protection
# frequency=500. # refreshing frequency (Hz)
# alpha = 1.05

class PISensor(motion.MotionSensor):
    def __init__(self, port='/dev/ttyS0', timeout=1, baudrate=9600, ser=None):
        ## @fn __init__()
        # @brief This class create an axis and opens the corresponding serial port.
        #
        # @param port_number : str
        #         Path to the corresponding serial port, e.g '/dev/ttyS4'
        # @param baud_rate : int, default = 38400
        #         Set the corresponding baud rate.
        # @param timeout : int or float, default = 1
        #         Serial timeout.
        super(PISensor, self).__init__(port, baudrate)
        self.port = port
        self.timeout = timeout

        if ser is not None:
            self.ser = ser
        else:
            self.ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout)
            a = self.ser.write("%c%cSA%d\r" % (
            1, '0', 10000))  # fixer acceleration de 10 000 a 100 000 microsteps par seconde au carre
            a = self.ser.write("%c%cSV%d\r" % (1, '0', 10000))  # fixer vitesse

    def get_position(self):
        self.ser.write("%c%cTP\r" % (1, '0'))
        return self.ser.readline()
