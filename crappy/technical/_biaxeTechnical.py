import serial
#from ..sensor import _biotensSensor
from ..actuator import _biaxeActuator

class Biaxe(object):
	def __init__(self, port='/dev/ttyUSB0'):
		self.port=port
		#self.ser=serial.Serial(self.port, baudrate=19200, timeout=0.1)
		self.sensor=None
		self.actuator=_biaxeActuator.BiaxeActuator(self.port)
		
		