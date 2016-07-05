# coding: utf-8
import serial
import time
from ..sensor import _biotensSensor
from ..actuator import _biotensActuator
from ._meta import motion


class Biotens(motion.Motion):
    def __init__(self, port='/dev/ttyUSB0', baudrate=19200, size=30):
        """
        Open the connection, and initialise the Biotens.

        You should always use this Class to communicate with the Biotens.

        Parameters
        ----------
        port : str, default = '/dev/ttyUSB0'
            Path to the correct serial port.
        size : int of float, default = 30
            Initial size of your test sample, in mm.
        """
        super(Biotens, self).__init__(port, baudrate)
        self.size = size - 7
        if self.size < 0:
            self.size = 0
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(self.port, baudrate=19200, timeout=0.1)
        self.actuator = _biotensActuator.BiotensActuator(self.ser, self.size)
        self.sensor = _biotensSensor.BiotensSensor(ser=self.ser)
        self.clear_errors()
        self.initialisation()
        self.mise_position()

    def mise_position(self):
        """Set motor into position for sample's placement"""
        self.actuator.set_position(self.size + 0.3, 70)  # add 0.2 to ensure going to the wanted position
        startposition = '\x52\x52\x52\xFF\x00' + _biotensSensor.convert_to_byte(10,
                                                                                'B') + _biotensSensor.convert_to_byte(4,
                                                                                                                      'B') + _biotensSensor.convert_to_byte(
            0, 'i') + '\xAA\xAA\x50\x50\x50\xFF\x00' + _biotensSensor.convert_to_byte(10, 'B') + '\xAA\xAA'
        self.ser.write(startposition)
        try:
            self.ser.readlines()
        except serial.SerialException:
            pass
        last_position_si = 0
        position_si = 99
        while position_si != last_position_si:
            last_position_si = position_si
            position_si = self.sensor.get_position()
            print "position : ", position_si
        print "Fin"
        self.stop()

    def initialisation(self):
        """Actuators goes out completely, in order to set the initial position"""

        init_position = '\x52\x52\x52\xFF\x00' + _biotensSensor.convert_to_byte(38,
                                                                               'B') + _biotensSensor.convert_to_byte(4,
                                                                                                                     'B') + _biotensSensor.convert_to_byte(
            0, 'i') + '\xAA\xAA\x50\x50\x50\xFF\x00' + _biotensSensor.convert_to_byte(38, 'B') + '\xAA\xAA'
        init_speed = '\x52\x52\x52\xFF\x00' + _biotensSensor.convert_to_byte(40, 'B') + _biotensSensor.convert_to_byte(2,
                                                                                                                      'B') + _biotensSensor.convert_to_byte(
            -50, 'h') + '\xAA\xAA\x50\x50\x50\xFF\x00' + _biotensSensor.convert_to_byte(40, 'B') + '\xAA\xAA'
        init_torque = '\x52\x52\x52\xFF\x00' + _biotensSensor.convert_to_byte(41, 'B') + _biotensSensor.convert_to_byte(
            2, 'B') + _biotensSensor.convert_to_byte(1023,
                                                     'i') + '\xAA\xAA\x50\x50\x50\xFF\x00' + _biotensSensor.convert_to_byte(
            41, 'B') + '\xAA\xAA'
        to_init = '\x52\x52\x52\xFF\x00' + _biotensSensor.convert_to_byte(37, 'B') + _biotensSensor.convert_to_byte(2,
                                                                                                                   'B') + _biotensSensor.convert_to_byte(
            0, 'h') + '\xAA\xAA\x50\x50\x50\xFF\x00' + _biotensSensor.convert_to_byte(37, 'B') + '\xAA\xAA'

        self.ser.writelines([init_position, init_speed, init_torque, to_init])
        self.ser.write(
            '\x52\x52\x52\xFF\x00' + _biotensSensor.convert_to_byte(2, 'B') + _biotensSensor.convert_to_byte(2,
                                                                                                             'B') + _biotensSensor.convert_to_byte(
                12, 'h') + '\xAA\xAA\x50\x50\x50\xFF\x00' + _biotensSensor.convert_to_byte(2, 'B') + '\xAA\xAA')
        last_position_si = 0
        position_si = 99
        time.sleep(1)
        while position_si != last_position_si:
            last_position_si = position_si
            position_si = self.sensor.get_position()
            print "position : ", position_si
        print "init done"
        self.stop()
        # time.sleep(1)
        # initializes the count when the motors is out.
        start_position = '\x52\x52\x52\xFF\x00' + _biotensSensor.convert_to_byte(10,
                                                                                'B') + _biotensSensor.convert_to_byte(4,
                                                                                                                      'B') + _biotensSensor.convert_to_byte(
            0, 'i') + '\xAA\xAA\x50\x50\x50\xFF\x00' + _biotensSensor.convert_to_byte(10, 'B') + '\xAA\xAA'
        self.ser.write(start_position)
        # time.sleep(1)
        try:
            self.ser.readlines()
        except serial.SerialException:
            pass

    def reset(self):
        # TODO
        pass

    def stop(self):
        """Stop the motor. Amazing."""
        command = '\x52\x52\x52\xFF\x00' + _biotensSensor.convert_to_byte(2, 'B') + _biotensSensor.convert_to_byte(2,
                                                                                                                   'B') + _biotensSensor.convert_to_byte(
            0, 'h') + '\xAA\xAA\x50\x50\x50\xFF\x00' + _biotensSensor.convert_to_byte(2, 'B') + '\xAA\xAA'
        self.ser.write(command)
        # return command

    def close(self):
        self.stop()
        self.ser.close()

    def clear_errors(self):
        """Clears error in motor registers. obviously."""
        command = '\x52\x52\x52\xFF\x00' + _biotensSensor.convert_to_byte(35, 'B') + _biotensSensor.convert_to_byte(4,
                                                                                                                    'B') + _biotensSensor.convert_to_byte(
            0, 'i') + '\xAA\xAA\x50\x50\x50\xFF\x00' + _biotensSensor.convert_to_byte(35, 'B') + '\xAA\xAA'
        self.ser.write(command)
