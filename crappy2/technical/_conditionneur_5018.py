# coding: utf-8
import serial


class Conditionner_5018:
    def __init__(self,port = '/dev/ttyS5', baudrate = 115200):
	self.baudrate = baudrate
        self.port = port
        self.ser = serial.Serial(self.port, baudrate=self.baudrate, parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)
        
        
    def reset(self):
	self.ser.write('9,0\r\n')
	self.ser.write('9,1\r\n')