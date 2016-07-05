#!/usr/bin/python
# -*- coding: utf-8 -*-
##  @addtogroup actuator
# @{

##  @defgroup BiotensActuator BiotensActuator
# @{

## @file _biotensActuator.py
# @brief  This class contains methods to command the motors of the biotens
#         machine. You should NOT use it directly, but use the BiotensTechnical.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 04/07/2016

from struct import *
import serial
import time
from ._meta import motion
from .._warnings import deprecated as deprecated


def convert_to_byte(number, length):
    """This functions converts decimal into bytes or bytes into decimals. 
    Mandatory in order to send or read anything into/from MAC Motors registers."""
    encoded = pack('%s' % length, number)  # get hex byte sequence in required '\xXX\xXX', big endian format.
    b = bytearray(encoded, 'hex')
    i = 0
    c = ''
    for i in range(0, len(encoded)):
        x = int(b[i]) ^ 0xff  # get the complement to 255
        x = pack('B', x)  # byte formalism
        c += encoded[i] + '%s' % x  # concatenate byte and complement and add it to the sequece
    return c


# -------------------------------------------------------------------------------------------
# This function allows to start the motor in desired mode (1=speed,2=position) or stop it (mode 0).


class BiotensActuator(motion.MotionActuator):
    def __init__(self, ser, size):
        # @fn __init__()
        # @brief This class contains methods to command the motors of the biotens
        # machine. You should NOT use it directly, but use the BiotensTechnical.
        # @param ser serial instance
        # @param size ???

        super(BiotensActuator, self).__init__()
        self.ser = ser
        self.size = size
        # self.clear_errors()

    def set_speed(self, speed):
        """Pilot in speed mode, requires speed in mm/min"""
        # converts speed in motors value
        # displacement rate in mm/min, V_SOll in 1/16 encoder counts/sample.
        # 4096 encounder counts/revolution, sampling frequency = 520.8Hz, screw thread=5.
        speed_soll = int(round(16 * 4096 * speed / (520.8 * 60 * 5)))
        set_speed = '\x52\x52\x52\xFF\x00' + convert_to_byte(5, 'B') + convert_to_byte(2, 'B') + convert_to_byte(
            speed_soll, 'h') + '\xAA\xAA\x50\x50\x50\xFF\x00' + convert_to_byte(5, 'B') + '\xAA\xAA'

        # set torque to default value 1023
        set_torque = '\x52\x52\x52\xFF\x00' + convert_to_byte(7, 'B') + convert_to_byte(2, 'B') + convert_to_byte(1023,
                                                                                                                  'h') + \
                     '\xAA\xAA\x50\x50\x50\xFF\x00' + convert_to_byte(7, 'B') + '\xAA\xAA'

        # set acceleration to 10000 mm/s² (default value, arbitrarily chosen, works great so far)
        asoll = int(round(16 * 4096 * 10000 / (520.8 * 520.8 * 5)))
        set_acceleration = '\x52\x52\x52\xFF\x00' + convert_to_byte(6, 'B') + convert_to_byte(2, 'B') + convert_to_byte(
            asoll, 'h') + '\xAA\xAA\x50\x50\x50\xFF\x00' + convert_to_byte(6, 'B') + '\xAA\xAA'

        command = '\x52\x52\x52\xFF\x00' + convert_to_byte(2, 'B') + convert_to_byte(2, 'B') + \
                  convert_to_byte(1, 'h') + '\xAA\xAA\x50\x50\x50\xFF\x00' + convert_to_byte(2, 'B') + '\xAA\xAA'

        # write every parameters in motor's registers
        self.ser.writelines([set_speed, set_torque, set_acceleration, command])

    def set_position(self, position, speed):
        """Pilot in position mode, needs speed and final position to run (in mm/min and mm)"""
        # conversion of position from mm into encoder's count
        position_soll = int(round(position * 4096 / 5))
        set_position = '\x52\x52\x52\xFF\x00' + convert_to_byte(3, 'B') + convert_to_byte(4, 'B') + convert_to_byte(
            position_soll, 'i') + '\xAA\xAA\x50\x50\x50\xFF\x00' + convert_to_byte(3, 'B') + '\xAA\xAA'

        # converts speed in motors value
        # displacement rate in mm/min, V_SOll in 1/16 encoder counts/sample.
        # 4096 encounder counts/revolution, sampling frequency = 520.8Hz, screw thread=5.
        speed_soll = int(round(16 * 4096 * speed / (520.8 * 60 * 5)))
        set_speed = '\x52\x52\x52\xFF\x00' + convert_to_byte(5, 'B') + convert_to_byte(2, 'B') + convert_to_byte(
            speed_soll, 'h') + '\xAA\xAA\x50\x50\x50\xFF\x00' + convert_to_byte(5, 'B') + '\xAA\xAA'

        # set torque to default value 1023
        set_torque = '\x52\x52\x52\xFF\x00' + convert_to_byte(7, 'B') + convert_to_byte(2, 'B') + \
                     convert_to_byte(1023, 'h') + '\xAA\xAA\x50\x50\x50\xFF\x00' + convert_to_byte(7, 'B') + '\xAA\xAA'

        # set acceleration to 10000 mm/s² (default value, arbitrarily chosen, works great so far)
        asoll = int(round(16 * 4096 * 10000 / (520.8 * 520.8 * 5)))
        set_acceleration = '\x52\x52\x52\xFF\x00' + convert_to_byte(6, 'B') + convert_to_byte(2, 'B') + convert_to_byte(
            asoll, 'h') + '\xAA\xAA\x50\x50\x50\xFF\x00' + convert_to_byte(6, 'B') + '\xAA\xAA'

        command = '\x52\x52\x52\xFF\x00' + convert_to_byte(2, 'B') + convert_to_byte(2, 'B') + \
                  convert_to_byte(2, 'h') + '\xAA\xAA\x50\x50\x50\xFF\x00' + convert_to_byte(2, 'B') + '\xAA\xAA'

        # write every parameters in motor's registers
        self.ser.writelines([set_position, set_speed, set_torque, set_acceleration, command])

    @deprecated(set_position)
    def setmode_position(self, position, speed):
        """
        DEPRECATED: use set_position instead.
        """
        self.set_position(position, speed)

    @deprecated(set_speed)
    def setmode_speed(self, speed):
        """
        DEPRECATED: use set_speed instead.
        """
        self.set_speed(speed)

    @deprecated(None, warn_msg="Stop method defined in _biotensTechnical")
    def stop_motor(self):
        """
        DEPRECATED: stop method defined in _biotensTechnical
        Stop the motor. Amazing.
        """
        command = '\x52\x52\x52\xFF\x00' + convert_to_byte(2, 'B') + convert_to_byte(2, 'B') + convert_to_byte(0, 'h') \
                  + '\xAA\xAA\x50\x50\x50\xFF\x00' + convert_to_byte(2, 'B') + '\xAA\xAA'
        self.ser.write(command)
        # return command

    @deprecated(None, "clear_errors defined in _biotensTechnical")
    def clear_errors(self):
        """
        DEPRECATED: clear_errors defined in _biotensTechnical.
        Clears error in motor registers. obviously.
        """
        command = '\x52\x52\x52\xFF\x00' + convert_to_byte(35, 'B') + convert_to_byte(4, 'B') + \
                  convert_to_byte(0, 'i') + '\xAA\xAA\x50\x50\x50\xFF\x00' + convert_to_byte(35, 'B') + '\xAA\xAA'
        self.ser.write(command)
