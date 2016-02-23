# coding: utf-8
import serial
import time
from ..sensor import _biotensSensor
from ..actuator import _biotensActuator

class Biotens(object):
	"""Open both a BiotensSensor and BiotensActuator instances."""
	def __init__(self, port='/dev/ttyUSB0', size=30):
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
		self.size=size-7
		if self.size<0:
			self.size=0
		self.port=port
		self.ser=serial.Serial(self.port, baudrate=19200, timeout=0.1)
		self.sensor=_biotensSensor.BiotensSensor(self.ser)
		self.actuator=_biotensActuator.BiotensActuator(self.ser, self.size)
		self.initialisation()
		self.mise_position()
	
	def mise_position(self): 
		"""Set motor into position for sample's placement"""
		#print "here"
		self.actuator.setmode_position(self.size+0.3,70) # add 0.2 to ensure going to the wanted position
		startposition='\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(10,'B')+_biotensSensor.convert_to_byte(4,'B')+_biotensSensor.convert_to_byte(0,'i')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(10,'B')+ '\xAA\xAA'
		self.ser.write(startposition)	
		try:
			self.ser.readlines()
		except serial.SerialException:
			pass
		last_position_SI=0
		position_SI=99
		while position_SI!=last_position_SI:
			last_position_SI=position_SI
			position_SI=self.sensor.read_position()
			print "position : ", position_SI
		print "Fin"
		self.actuator.stop_motor()	
		
		
	def initialisation(self): 
		"""Actuators goes out completely, in order to set the initial position"""
		
		initposition= '\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(38,'B')+_biotensSensor.convert_to_byte(4,'B')+_biotensSensor.convert_to_byte(0,'i')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(38,'B')+ '\xAA\xAA'
		initspeed = '\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(40,'B')+_biotensSensor.convert_to_byte(2,'B')+_biotensSensor.convert_to_byte(-50,'h')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(40,'B')+ '\xAA\xAA'
		inittorque = '\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(41,'B')+_biotensSensor.convert_to_byte(2,'B')+_biotensSensor.convert_to_byte(1023,'i')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(41,'B')+ '\xAA\xAA'	
		toinit= '\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(37,'B')+_biotensSensor.convert_to_byte(2,'B')+_biotensSensor.convert_to_byte(0,'h')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(37,'B')+ '\xAA\xAA'
		
		self.ser.writelines([initposition, initspeed, inittorque, toinit])
		self.ser.write('\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(2,'B')+_biotensSensor.convert_to_byte(2,'B')+_biotensSensor.convert_to_byte(12,'h')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(2,'B')+ '\xAA\xAA')
		last_position_SI=0
		position_SI=99
		time.sleep(1)
		while position_SI!=last_position_SI:
			last_position_SI=position_SI
			position_SI=self.sensor.read_position()
			print "position : ", position_SI
		print "init done"
		self.actuator.stop_motor()	
		#time.sleep(1)
		### initializes the count when the motors is out.
		startposition='\x52\x52\x52\xFF\x00'+_biotensSensor.convert_to_byte(10,'B')+_biotensSensor.convert_to_byte(4,'B')+_biotensSensor.convert_to_byte(0,'i')+'\xAA\xAA\x50\x50\x50\xFF\x00' +_biotensSensor.convert_to_byte(10,'B')+ '\xAA\xAA'
		self.ser.write(startposition)
		#time.sleep(1)
		try:
			self.ser.readlines()
		except serial.SerialException:
			pass
		