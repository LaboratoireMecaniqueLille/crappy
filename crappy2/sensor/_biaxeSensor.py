# coding: utf-8
#
##  @addtogroup sensor
# @{

##  @defgroup biaxe BiaxeSensor
# @{

## @file _biaxeSensor.py
# @brief  Declare a new axis for the Biaxe
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 21/06/2016

import serial
from ._meta import motion


class BiaxeSensor(motion.MotionSensor):
    """Declare a new axis for the Biaxe"""
    def __init__(self, port='/dev/ttyUSB0', baudrate=38400, timeout=1, ser=None):
        """
        This class create an axis and opens the corresponding serial port.

        Args:
            port : Path to the corresponding serial port, e.g '/dev/ttyS4'
            baudrate : Set the corresponding baud rate.
            timeout : Serial timeout.
        """
        super(BiaxeSensor, self).__init__(port, baudrate)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        if ser is not None:
            self.ser = ser
        else:
            self.ser = serial.Serial(self.port, self.baudrate,
                                     serial.EIGHTBITS, serial.PARITY_EVEN
                                     , serial.STOPBITS_ONE, self.timeout)
            self.ser.write("OPMODE 0\r\n EN\r\n")

    def get_position(self):
        """
        TODO
        Search for the physical position of the motor
        """
        pass
# @}
# @}
