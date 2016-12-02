# coding: utf-8
##  @addtogroup sensor
# @{

##  @defgroup Agilent34420ASensor Agilent34420ASensor
# @{

## @file _Agilent34420ASensor.py
# @brief  Sensor class for Agilent34420A devices.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 29/06/2016

import serial
from ._meta import acquisition


class Agilent34420ASensor(acquisition.Acquisition):
    """Sensor class for Agilent34420A devices."""

    def __init__(self, mode="VOLT", device='/dev/ttyUSB0', baudrate=9600, timeout=10):
        """
        This class contains method to measure values of resistance or tension on Agilent34420A devices.

        May work for other devices too, but not tested.
        If you have issues with this class returning a lot of 'bad serial', \
        make sure you have the last version of pySerial.

        Args:
            mode : {"VOLT","RES"} , default = "VOLT"
                    Desired value to measure.
            device : str, default = '/dev/ttyUSB0'
                    Path to the device.
            baudrate : int, default = 9600
                    Desired baudrate.
            timeout : int or float, default = 10
                    Timeout for the serial connection.
        """
        super(Agilent34420ASensor, self).__init__()
        ## path to the device
        self.device = device
        ## desired baudrate
        self.baudrate = baudrate
        ## timeout for the serial connection
        self.timeout = timeout
        ## desired value to measure
        self.mode = mode
        ## Serial instance
        self.ser = serial.Serial(port=self.device, baudrate=self.baudrate, timeout=self.timeout)
        self.new()

    def new(self):
        self.ser.write("*RST;*CLS;*OPC?\n")
        self.ser.write("SENS:FUNC \"" + self.mode + "\";  \n")
        self.ser.write("SENS:" + self.mode + ":NPLC 2  \n")
        # ser.readline()
        self.ser.write("SYST:REM\n")
        self.get_data()

    def get_data(self):
        """
        Read the signal, return False if error and print 'bad serial'.
        """
        try:
            self.ser.write("READ?  \n")
            # tmp = self.ser.readline()
            tmp = self.ser.read(self.ser.in_waiting)
            self.ser.flush()
            # print tmp
            return float(tmp)
        except Exception as e:
            print e
            # self.ser.read(self.ser.inWaiting())
            # print self.ser.inWaiting()
            # self.ser.flush()
            # time.sleep(0.5)
            return False

    def close(self):
        """
        Close the serial port.
        """
        self.ser.close()
# @}
# @}
