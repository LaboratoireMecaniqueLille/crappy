#!/usr/bin/python
# -*- coding: utf-8 -*-
# import time
from struct import *
import serial


# This functions converts decimal into bytes or bytes into decimals.
#  Mandatory in order to send or read anything into/from MAC Motors registers.

def convert_to_byte(number, length):
    """
    This functions converts decimal into bytes.  Mandatory in order to send
    or read anything into/from MAC Motors registers."""
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
    or read anything into/from MAC Motors registers."""
    # sequence=sequence[::2] ## cut off "complement byte"
    decim = unpack('i', sequence)  # convert to signed int value
    return decim[0]


# -------------------------------------------------------------------------------------------
# This function allows to start the motor in desired mode (1=velocity,2=position) or stop it (mode 0).


class BiotensSensor(object):
    def __init__(self, ser):
        """
        This class contains methods to get info from the motors of the biotens
        machine. You should NOT use it directly, but use the BiotensTechnical.
        """
        self.ser = ser

    def read_position(self):
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

    # ajouter d'autre fonctions si besoin (Thanks captain obvious)
