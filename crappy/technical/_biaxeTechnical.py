# coding: utf-8
import serial
#from ..sensor import _biotensSensor
from ..actuator import _biaxeActuator

class Biaxe(object):
	def __init__(self, port='/dev/ttyUSB0'):
		"""Opens a BiaxActuator instance. No BiaxeSensor at the moment."""
		self.port=port
		#self.ser=serial.Serial(self.port, baudrate=19200, timeout=0.1)
		self.sensor=None
		self.actuator=_biaxeActuator.BiaxeActuator(self.port)
		
	def close(self):
		"""Set speed to 0 and close serial connection"""
		self.actuator.set_speed(0)
		self.actuator.close_port()
		