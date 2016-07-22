#!/usr/bin/python
# -*- coding: utf-8 -*-
##  @addtogroup sensor
# @{

##  @defgroup OrientalSensor OrientalSensor
# @{

## @file _orientalSensor.py
# @brief  Sensor class for oriental motors.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 29/06/2016

import serial
from ._meta import motion


class OrientalSensor(motion.MotionSensor):
    def __init__(self, baudrate=115200, port='/dev/ttyUSB0', num_device=1, conversion_factor=1, ser=None):
        """

        Args:
            baudrate: wanted baudrate to configure port communication
            port: Path to the corresponding serial port, e.g '/dev/ttyUSB0'
            num_device: number of the device
            conversion_factor: factor to convert received data to a more physical value.
            ser: serial instance, use it in case you have already initialized the serial port with this motor,
                for example, if you use the the OrientalSensor and the OrientalActuator (in this case you should use
                OrientalTechnical).
        Returns:
            OrientalSensor instance.
        """
        super(OrientalSensor, self).__init__(port, baudrate)
        ## wanted baudrate to configure port communication
        self.num_device = num_device
        ## number of the device
        self.baudrate = baudrate
        ## Path to the corresponding serial port, e.g '/dev/ttyUSB0'
        self.port = port
        ## factor to convert received data to a more physical value.
        self.conversion_factor = conversion_factor

        # Actuator _ Declaration
        try:
            if ser is not None:
                ## serial instance
                self.ser = ser
            else:
                ## serial instance
                self.ser = serial.Serial(self.port)
                self.ser.timeout = 0.01
                self.ser.baudrate = self.baudrate
                self.ser.bytesize = 8
                self.ser.stopbits = 1
                self.ser.parity = 'N'
                self.ser.xonxoff = False
                self.ser.rtscts = False
                self.ser.dsrdtr = False
                self.ser.close()
                self.ser.open()
                for i in range(4):
                    self.ser.write("TALK{0}\n".format(i+1))
                    ret=self.ser.readlines()
                    if "{0}>".format(i+1) in ret:
                        self.num_device = i+1
                        motors = ['A', 'B', 'C', 'D']
                        print "Motor connected to port {0} is {1}".format(self.port, motors[i])
                        break
        except Exception as e:
            print e

    def write_cmd(self, cmd):
        """
        Format a command and write it to the serial port communication.

        Prints out the output until there is no data
        available.

        Args:
            cmd: ASCII command to write on the serial port.

        Returns:
            void return function, it just prints out data read on the serial port.
        """
        self.ser.write("{0}\n".format(cmd))
        ret = self.ser.readline()
        # while ret != '{0}>'.format(self.num_device):
        while ret != '' and ret != '{0}>'.format(self.num_device):
            print ret
            ret = self.ser.readline()

    def get_position(self):
        """
        Return the current position of the motor.

        Returns:
            current position of the motor (float).
        """
        self.ser.flushInput()
        self.ser.write('PC\n')
        a_jeter = self.ser.readline()
        actuator_position = self.ser.readline()
        # self.ser.close()
        actuator_position = str(actuator_position)
        actuator_position = actuator_position[4::]
        actuator_position = actuator_position[::-1]
        actuator_position = actuator_position[3::]
        actuator_position = actuator_position[::-1]
        try:
            actuator_position = float(actuator_position) * self.conversion_factor
            return actuator_position
        except ValueError:
            print "PositionReadingError"
