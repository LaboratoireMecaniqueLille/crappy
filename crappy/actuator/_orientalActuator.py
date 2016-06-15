#!/usr/bin/python
# -*- coding: utf-8 -*-
import time

from crappy.sensor._orientalSensor import convert_to_byte


class OrientalActuator(object):
    def __init__(self, ser, size):
        """This class contains methods to command the motors of the biotens
        machine. You should NOT use it directly, but use the BiotensTechnical.
        """
        self.size = size
        self.ser = ser
        self.clear_errors()

    def stop_motor(self):
        """Stop the motor. Amazing."""
        command = 'SSTOP\n'
        self.ser.write(command)

    # return command

    def setmode_position(self, position, speed):
        """
        Pilot in position mode, needs speed and final position to run (in mm/min and mm)
        """
        # conversion of position from mm into encoder's count

    def setmode_speed(self, speed):
        """Pilot in speed mode, requires speed in mm/min"""
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
                  convert_to_byte(1, 'h') + '\xAA\xAA\x50\x50\x50\xFF\x00' + convert_to_byte(2, 'B') + '\xAA\xAA'

        # write every parameters in motor's registers
        self.ser.writelines([set_speed, set_torque, set_acceleration, command])

    # def mise_position(self): # set motor into position for sample's placement
    # self.setmode_position(self.size,70)
    # startposition = '\x52\x52\x52\xFF\x00' + convert_to_byte(10, 'B') + convert_to_byte(4, 'B') + \
    #                 convert_to_byte(0,'i') + '\xAA\xAA\x50\x50\x50\xFF\x00' + convert_to_byte(10, 'B') + '\xAA\xAA'
    # self.ser.write(startposition)
    # self.ser.readlines()
    # position_SI=0.
    # while position_SI-self.size-1.<=0.:
    # position_= Out(10).get_command() #command sequence for reading register 10 = actual position
    # position=ser.write(position_)
    # position2=ser.read(19)
    # position_=GetAnswer(position2).get_answer()

    # position_SI=Conversion(position_).displacement_SI()

    # self.stop_motor()

    def clear_errors(self):
        """Clears error in motor registers. obviously."""
        command = '\x52\x52\x52\xFF\x00' + convert_to_byte(35, 'B') + convert_to_byte(4, 'B') + \
                  convert_to_byte(0,'i') + '\xAA\xAA\x50\x50\x50\xFF\x00' + convert_to_byte(35, 'B') + '\xAA\xAA'
        self.ser.write(command)
