import serial
from ..sensor import _biotensSensor
from ..actuator import _biotensActuator

class Biotens(object):
	def __init__(self, port='/dev/ttyUSB0', size=30):
		self.size=size
		self.port=port
		self.ser=serial.Serial(self.port, baudrate=19200, timeout=0.1)
		self.sensor=_biotensSensor.BiotensSensor(self.ser)
		self.actuator=_biotensActuator.BiotensActuator(self.ser, self.size)
		
		