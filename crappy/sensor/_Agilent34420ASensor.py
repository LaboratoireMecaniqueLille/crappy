class Agilent34420ASensor(object):
	"""
	Sensor class for Agilent34420A devices.
	"""
	def __init__(self,device='/dev/ttyUSB0',baud_rate=9600,timeout=1):
		# We could add several type of measure: RES, CUR, TENS...
		import serial
		#self.serial=serial
		self.device=device
		self.baud_rate=baud_rate
		self.timeout=timeout
		ser = serial.Serial(self.device,self.baud_rate,self.timeout)
		ser.write("*RST;*CLS;*OPC?\n")
		ser.write("SENS:FUNC \"RES\";  \n")
		ser.write("SENS:RES:NPLC 2  \n")
		ser.readline()
		ser.write("SYST:REM\n")
		

	def getData(self):
		"""Read the signal, return False if error"""
		try:
			self.ser.write("READ?  \n")
			tmp = ser.readline()
			self.ser.flush()
			return float(tmp)
		except:
			print "bad serial"
			return False
					
	def close(self):
		self.ser.close()
		
