# coding: utf-8
##  @addtogroup actuator
# @{

##  @defgroup Agilent34420AActuator Agilent34420AActuator
# @{

## @file _Agilent34420AActuator.py
# @brief  Actuator class for Agilent34420A devices.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 29/06/2016

import serial
from ._meta import command


class Agilent34420AActuator(command.Command):
    """Actuator class for Agilent34420A devices."""

    def __init__(self, subdevice, channel, range_num, gain, offset, mode="VOLT", device='/dev/ttyUSB0', baudrate=9600,
                 timeout=10):
        ## @fn __init__()
        # @brief If you have issues with this class returning a lot of 'bad serial', \
        # make sure you have the last version of pySerial.
        #
        # @param mode : {"VOLT","RES"} , default = "VOLT"
        #         Desired value to measure.
        # @param device : str, default = '/dev/ttyUSB0'
        #         Path to the device.
        # @param baudrate : int, default = 9600
        #         Desired baudrate.
        # @param timeout : int or float, default = 10
        #         Timeout for the serial connection.
        super(Agilent34420AActuator, self).__init__(device, subdevice, channel, range_num, gain, offset)
        self.device = device
        self.baudrate = baudrate
        self.timeout = timeout
        self.mode = mode
        self.ser = serial.Serial(port=self.device, baudrate=self.baudrate, timeout=self.timeout)
        self.new()

    def new(self):
        self.ser.write("*RST;*CLS;*OPC?\n")
        self.ser.write("SENS:FUNC \"" + self.mode + "\";  \n")
        self.ser.write("SENS:" + self.mode + ":NPLC 2  \n")
        # ser.readline()
        self.ser.write("SYST:REM\n")

    def set_cmd(self):
        """
        TODO
        """
        pass

    def close(self):
        """
        Close the serial port.
        """
        self.ser.close()
