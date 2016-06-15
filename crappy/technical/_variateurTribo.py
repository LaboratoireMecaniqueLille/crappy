# coding: utf-8
import serial
from ..sensor import _variateurTriboSensor
from ..actuator import _variateurTriboActuator
from ._meta import motion


class VariateurTribo(motion.Motion):
    def __init__(self, port='/dev/ttyS0', port_arduino='/dev/ttyACM0', baudrate=38400):
        self.baudrate = baudrate
        self.port = port
        self.port_arduino = port_arduino
        self.ser = serial.Serial(self.port, baudrate=self.baudrate, parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
        self.ser_arduino = serial.Serial(self.port_arduino, baudrate=9600, parity=serial.PARITY_NONE,
                                         stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
        self.sensor = _variateurTriboSensor.VariateurTriboSensor(self.ser)
        self.actuator = _variateurTriboActuator.VariateurTriboActuator(ser_servostar=self.ser,
                                                                       ser_arduino=self.ser_arduino)

    def reset(self):
        """
        TODO
        """
        pass

    def stop(self):
        """
        stop motor
        """
        self.ser.write('dis\r\n')

    def close(self):
        self.ser.close()
        self.ser_arduino.close()

    def clear_errors(self):
        while self.ser.inWaiting() > 0:
            self.ser.read(1)
