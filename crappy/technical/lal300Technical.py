import time
import serial
from serial import SerialException
from ..sensor import SensorLal300
from ..actuator import ActuatorLal300

class TechnicalLal300(object):
	
	def __init__(self,param): # add param as parameter
		self.param=param
		self.ser=serial.Serial(port=param['port'], #Parametre initiaux du port serie
		baudrate=param['baudrate'],
		bytesize=serial.EIGHTBITS,
		parity=serial.PARITY_NONE,
		stopbits=serial.STOPBITS_ONE,
		timeout=param['timeout'],
		rtscts=False,
		writeTimeout=None,
		dsrdtr=False,
		interCharTimeout=None)
		self.actuator=ActuatorLal300(self.param,self.ser) #appel de la sous-classe actionneur
		self.sensor=SensorLal300(self.param,self.ser) #appel de la sous-classe capteur
		self.pos=self.sensor.checkdisp()
		
		
        def set_position_checkdisp(self,consigne): #Actionne le moteur en mode position
		last_pos=self.sensor.checkdisp()
		self.actuator.set_position(consigne)
		self.pos=self.sensor.checkdisp()
		while self.pos <= (last_pos -200) or self.pos >= (last_pos + 200):
			last_pos=self.pos
			self.pos=self.sensor.checkdisp()
			
	# def set_position_main(self):#Actionne le moteur en mode position
		# try:
			# t0=time.time()
			# #for i in range(20):
			# for i in range(self.param['CYCLES']):
				# try:
					# TIME=time.time()-t0
					# cycles=i
					# self.actuator.set_position("MN,PM,SA%i,SV%i,SQ%i,MA%i,GO\r\n"%(self.param['ACC'],self.param['SPEED'],self.param['FORCE'],self.param['POSITION']))
					# DISPLal300=self.sensor.checkdisp()
					# while DISPLal300 <= (self.param['POSITION']+self.param['CORR-']) or DISPLal300 >= (self.param['POSITION']+self.param['CORR+']):
						# DISPLal300=self.sensor.checkdisp()
						# TIME=time.time()-t0
						# self.save(cycles,TIME,DISPLal300)

					# self.actuator.set_position("MN,PM,SA%i,SV%i,SQ%i,MA%i,GO\r\n"%(self.param['ACC'],self.param['SPEED'],self.param['FORCE'],self.param['ORIGIN']))
					# DISPLal300=self.sensor.checkdisp()
					# while DISPLal300 <= (self.param['ORIGIN']+self.param['CORR-']) or DISPLal300 >= (self.param['ORIGIN']+self.param['CORR+']):
						# DISPLal300=self.sensor.checkdisp()
						# TIME=time.time()-t0
						# self.save(cycles,TIME,DISPLal300)
						# TIME=time.time()-t0
		
					# print "Nombre de cycles:",cycles, "Temps:",int(TIME)
					
				# except serial.SerialException as s:
					# print "SerialException detectee: ",s
					# pass

			# self.actuator.stoplal300()
			# self.actuator.reset()

			# return self.ser
		
		# except KeyboardInterrupt as k:
				# self.ser.read(self.ser.inWaiting())
				# time.sleep(0.1)
				# self.ser.write("MN,PM,SA%i,SV%i,SQ%i,MA%i,GO\r\n"%(self.param['ACC'],self.param['SPEED'],self.param['FORCE'],self.param['ORIGIN']))
				# time.sleep(2)
				# self.actuator.stoplal300()
				# print " Erreur. Le programme s'est arrete avant la fin du nombre de cycles: ",k
				# time.sleep(0.5)
				# self.actuator.reset()
		
		# except Exception as e:
			# self.ser.read(self.ser.inWaiting())
			# time.sleep(0.1)
			# self.ser.write("MN,PM,SA%i,SV%i,SQ%i,MA%i,GO\r\n"%(self.param['ACC'],self.param['SPEED'],self.param['FORCE'],self.param['ORIGIN']))
			# time.sleep(2)
			# self.actuator.stoplal300()
			# print " Exception detectee: ",e
			# time.sleep(0.5)
			# self.actuator.reset()

		# except ValueError as v:
			# self.ser.read(self.ser.inWaiting())
			# time.sleep(0.1)
			# self.ser.write("MN,PM,SA%i,SV%i,SQ%i,MA%i,GO\r\n"%(self.param['ACC'],self.param['SPEED'],self.param['FORCE'],self.param['ORIGIN']))
			# time.sleep(2)
			# self.actuator.stoplal300()
			# print " ValueError detectee: ",v
			# time.sleep(0.5)
			# self.actuator.reset()
		
