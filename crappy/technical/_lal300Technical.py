# coding: utf-8
#import time
import serial
from serial import SerialException
from ..sensor import SensorLal300
from ..actuator import ActuatorLal300

class TechnicalLal300(object):
	"""Open both a Lal300Sensor and Lal300Actuator instances."""
	def __init__(self,param):
		"""
Open the connection, and initialise the Lal300.

You should always use this Class to communicate with the Lal300.

Parameters
----------
param : dict
	Dict of parameters.
	
		* 'port' : str
			Path to the serial port.
		* 'baudrate' : int
			Corresponding baudrate.
		* 'timeout' : float
			Timeout of the serial connection.
		* 'PID_PROP' : float
			Proportionnal coefficient of the PID.
		* 'PID_INT' : float
			Integral coefficient for the PID.
		* 'PID_DERIV' : float
			Derivative coefficient for the PID.
		* 'PID_INTLIM' : float
			Limit of the integral coefficient.
		* 'ACC' float
			Acceleration of the motor.
		* 'ACconv' : float 
			Conversion ACC values to mm/s/s
		* 'FORCE' : float 
			Maximal force provided by the motor.
		* 'SPEEDconv' : float 
			Conversion SPEED values to mm/s
		* 'ENTREE_VERIN' : str
			'DI1'
		* 'SORTIE_VERIN' : str 
			'DI0'
		* 'ETIRE': list of int
			List of extreme values for the position in traction.
		* 'COMPRIME': list of int
			List of extreme values for the position in compression.
		* 'SPEED' : list of int
			List of speed, for each group of cycles.
		* 'CYCLES' : list of int
			List of cycles, for each group.


Examples
--------
>>> param = {}
param['port'] = '/dev/ttyUSB1'
param['baudrate'] = 19200
param['timeout'] = 0.#s
n = 3 # modify with great caution
param['PID_PROP'] = 8/n
param['PID_INT'] = 30/n
param['PID_DERIV'] = 200/n
param['PID_INTLIM'] = 1000/n
param['ACC'] = 6000.
param['ACconv'] = 26.22#conversion ACC values to mm/s/s
param['FORCE'] =30000.
param['SPEEDconv'] = 131072.#conversion SPEED values to mm/s
param['ENTREE_VERIN']='DI1'
param['SORTIE_VERIN']='DI0'
##### modifiable values :
param['ETIRE']=[-900,-1000,-1100,-1200,-2400,-3600,-4800,-6000,-7200,-8400,-9800,-11000,-12000,-18000,-24000,-36000,-48000,-60000,-72000,-84000]
param['COMPRIME']=[-200,-300,-400,-500,-700,-800,-900,-900,-1500,-3000,-3000,-5000,-5000,-5000,-5000,-5000,-5000,-5000,-5000,-5000]
param['SPEED'] = [15000,15000,15000,16000,30000,45000,80000,110000,130000,150000,180000,210000,250000,300000,350000,400000,500000,550000,600000,650000]
param['CYCLES']=[2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000,2000]
		"""
		self.param=param
		self.ser=serial.Serial(port=param['port'], #Configuration du port serie à l'aide de PySerial
		baudrate=param['baudrate'],
		bytesize=serial.EIGHTBITS,
		parity=serial.PARITY_NONE,
		stopbits=serial.STOPBITS_ONE,
		timeout=param['timeout'],
		rtscts=False,
		write_timeout=None,
		dsrdtr=False,
		inter_byte_timeout=None)
		self.actuator=ActuatorLal300(self.param,self.ser) #Appel de la sous-classe ActuatorLal300 avec les parametres situes dans le programme lal300Main.py
		self.sensor=SensorLal300(self.param,self.ser)  #Appel de la sous-classe SensorLal300 avec les parametres situes dans le programme lal300Main.py