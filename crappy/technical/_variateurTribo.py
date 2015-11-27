import serial
from ..sensor import _variateurTriboSensor
from ..actuator import _variateurTriboActuator

class VariateurTribo(object):
	def __init__(self, port='/dev/ttyS0'):
		self.port=port
		self.ser=serial.Serial(self.port, baudrate=38400, timeout=0.1)
		self.sensor=_variateurTriboSensor.variateurTriboSensor(self.ser)
		self.actuator=_variateurTriboActuator.variateurTriboActuator(self.ser)
