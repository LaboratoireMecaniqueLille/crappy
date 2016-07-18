#!/usr/bin/python
# -*- coding: utf-8 -*-
## @addtogroup sensor
# @{

##  @defgroup biotens BiotensSensor
# @{

## @file _biotensSensor.py
# @brief   Get info from biotens sensor
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 21/06/2016

from struct import *
import serial
from ._meta import motion
from .._warnings import deprecated as deprecated


def convert_to_byte(number, length):
    """
    This functions converts decimal into bytes.  Mandatory in order to send
    or read anything into/from MAC Motors registers.
    """
    encoded = pack('%s' % (length), number)  # get hex byte sequence in required '\xXX\xXX', big endian format.
    b = bytearray(encoded, 'hex')
    i = 0
    c = ''
    for i in range(0, len(encoded)):
        x = int(b[i]) ^ 0xff  # get the complement to 255
        x = pack('B', x)  # byte formalism
        c += encoded[i] + '%s' % x  # concatenate byte and complement and add it to the sequece
    return c


def convert_to_dec(sequence):
    """
    This functions converts bytes into decimals.  Mandatory in order to send
    or read anything into/from MAC Motors registers.
    """
    # sequence=sequence[::2] ## cut off "complement byte"
    decim = unpack('i', sequence)  # convert to signed int value
    return decim[0]


# -------------------------------------------------------------------------------------------
# This function allows to start the motor in desired mode (1=velocity,2=position) or stop it (mode 0).


class BiotensSensor(motion.MotionSensor):
    def __init__(self, ser=None, port='/dev/ttyUSB0', baudrate=19200):
        """
        This class contains methods to get info from the motors of the biotens
        machine. You should NOT use it directly, but use the BiotensTechnical.
        """

        super(BiotensSensor, self).__init__(port, baudrate)
        self.port = port
        self.baudrate = baudrate
        if ser is not None:
            self.ser = ser
        else:
            self.ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=0.1)

    def get_position(self):
        """Reads current position"""
        # print "reading position..."
        # print self.ser.inWaiting()
        try:
            self.ser.readlines()
        except serial.SerialException:
            # print "readlines failed"
            pass
        # print "position read"
        command = '\x50\x50\x50\xFF\x00' + convert_to_byte(10, 'B') + '\xAA\xAA'

        self.ser.write(command)
        # time.sleep(0.01)
        # print "reading..."
        # print self.ser.inWaiting()
        position_ = self.ser.read(19)
        # print "read"
        position = position_[9:len(position_) - 2:2]
        position = convert_to_dec(position) * 5 / 4096.
        return position

    @deprecated(get_position)
    def read_position(self):
        """
        DEPRECATED: Use get_position instead.
        """
        self.get_position()
# @}
# @}
