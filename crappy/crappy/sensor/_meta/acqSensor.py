import abc

class AcqSensor:
	
	__metaclass__= abc.ABCMeta

	@abc.abstractmethod
	def __init__(self):
		 pass
	 
	#@abc.abstractmethod
	#def new(self):
		#pass

	@abc.abstractmethod
	def getData(self):
		pass
			
	@abc.abstractmethod
	def close(self):
		pass