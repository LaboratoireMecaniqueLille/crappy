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

from struct import *
import serial

# This functions converts decimal into bytes or bytes into decimals.
#  Mandatory in order to send or read anything into/from MAC Motors registers.
from ._meta import motion


#
# def convert_to_byte(number, length):
#     """
#     This functions converts decimal into bytes.  Mandatory in order to send
#     or read anything into/from MAC Motors registers."""
#     encoded = pack('%s' % (length), number)  # get hex byte sequence in required '\xXX\xXX', big endian format.
#     b = bytearray(encoded, 'hex')
#     i = 0
#     c = ''
#     for i in range(0, len(encoded)):
#         x = int(b[i]) ^ 0xff  # get the complement to 255
#         x = pack('B', x)  # byte formalism
#         c += encoded[i] + '%s' % x  # concatenate byte and complement and add it to the sequece
#     return c
#
#
# def convert_to_dec(sequence):
#     """
#     This functions converts bytes into decimals.  Mandatory in order to send
#     or read anything into/from MAC Motors registers."""
#     # sequence=sequence[::2] ## cut off "complement byte"
#     decim = unpack('i', sequence)  # convert to signed int value
#     return decim[0]
#
#
# # -------------------------------------------------------------------------------------------
# # This function allows to start the motor in desired mode (1=velocity,2=position) or stop it (mode 0).


class OrientalSensor(motion.MotionSensor):
    def __init__(self, baudrate=115200, port='/dev/ttyUSB0', num_device=1, conversion_factor=1, ser=None):
        super(OrientalSensor, self).__init__(port, baudrate)
        self.num_device = num_device
        self.baudrate = baudrate
        self.port = port
        # Actuator _ Declaration
        try:
            if ser is not None:
                self.ser = ser
            else:
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
        self.conversion_factor = conversion_factor

    def write_cmd(self, cmd):
        self.ser.write("{0}\n".format(cmd))
        ret = self.ser.readline()
        # while ret != '{0}>'.format(self.num_device):
        while ret != '' and ret != '{0}>'.format(self.num_device):
            print ret
            ret = self.ser.readline()

    def get_position(self):
        # self.ser.open()
        self.ser.flushInput()
        self.ser.write('PC\n')
        a_jeter = self.ser.readline()
        ActuatorPos = self.ser.readline()
        # self.ser.close()
        ActuatorPos = str(ActuatorPos)
        ActuatorPos = ActuatorPos[4::]
        ActuatorPos = ActuatorPos[::-1]
        ActuatorPos = ActuatorPos[3::]
        ActuatorPos = ActuatorPos[::-1]
        try:
            ActuatorPos = float(ActuatorPos) * self.conversion_factor
            return ActuatorPos
        except ValueError:
            print "PositionReadingError"
