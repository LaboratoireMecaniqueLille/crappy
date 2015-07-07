import serial
from ..sensor import _biotensSensor
from ..actuator import _biotensActuator

class Biotens(object):
	def __init__(self, port, size):
		self.size=size
		self.ser=serial.Serial(port, baudrate=19200, timeout=0.1)
		self.sensor=_biotensSensor.BiotensSensor(self.ser)
		self.actuator=_biotensActuator.BiotensActuator(self.ser, self.size)
    
    