# coding: utf-8
import serial
from ..sensor import _variateurTriboSensor
from ..actuator import _variateurTriboActuator

class VariateurTribo(object):
	def __init__(self, port='/dev/ttyS0',port_arduino='/dev/ttyACM0'):
		self.port=port
		self.port_arduino=port_arduino
		self.ser=serial.Serial(self.port, baudrate=9600,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,bytesize=serial.EIGHTBITS)
		self.ser_arduino=serial.Serial(self.port_arduino,baudrate=9600,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,bytesize=serial.EIGHTBITS)
		self.sensor=_variateurTriboSensor.VariateurTriboSensor(self.ser)
		self.actuator=_variateurTriboActuator.VariateurTriboActuator(self.ser,self.ser_arduino)
