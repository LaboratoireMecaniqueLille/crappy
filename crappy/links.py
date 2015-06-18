from multiprocessing import Pipe
import abc

class Link_old:
	"""
[deprecated] Main class for links. All links should inherit this class.
	"""
	
	
	def __init__(self,trigger=None,condition=None,boolean=False):
		"""
Link([trigger=None,][condition=None,][boolean=False])

Creates a pipe with trigger and a condition as attributes, and is used to 
transfer information between blocks using a pipe, triggered by the condition.

Parameters
----------
trigger : string, optional
	The type of the trigger. Can be 'internal' , 'external' or None (default). 
	If None, sends everything.
condition : function or Pipe, optionnal
	The condition to set True the trigger. 
	If trigger is 'internal', condition should be a class taking the inside 
	value as parameter of a method evaluate.
	If trigger is 'external', condition should be a Link sending booleans.
boolean: Boolean, optionnal
	If True, send the boolean value of the condition instead of the input 
	value.
	
Attributes
----------
in_ : input extremity of the pipe.
out_ : output extremity of the pipe.
trigger : type of trigger ('internal', 'external' or None)
condition : class or Link object
boolean : Boolean type

Methods
-------
send(pickable) : sends a pickable object (or the boolean returned by the 
	condition).
recv(pickable) : receive a pickable object.
		"""
		
		self.out_,self.in_=Pipe()
		self.trigger=trigger
		self.condition=condition
		self.boolean=boolean
	
	def send(self,value):
		if self.trigger==None:
			if self.boolean:
				self.out_.send(True) 
			else:
				self.out_.send(value) 
		elif self.trigger=='internal':
			bool_condition, ret_value =self.condition.evaluate(value)
			#print bool_condition
			#print ret_value
			if bool_condition:
				if self.boolean:
					self.out_.send(True) 
				else:
					self.out_.send(ret_value)   
		elif self.trigger=='external': # for external trigger
			if self.condition.in_.poll(): 
				# if input pipe is readable
				val=self.condition.in_.recv() # read pipe
				if val: #if condition is True :
						if self.boolean:
							self.out_.send(True) 
						else:
							self.out_.send(value)
	
	def recv(self):
		return self.in_.recv()
	
	
class Link:
	"""
Main class for links. All links should inherit this class.
	"""
	
	
	def __init__(self,condition=None):
		"""
Link([condition=None])

Creates a pipe with a condition as attribute, and is used to transfer 
information between blocks using a pipe, triggered by the condition.

Parameters
----------
condition : Children class of links.Condition, optionnal
	Each "send" call will pass through the condition.evaluate method and sends
	the returned value.
	
Attributes
----------
in_ : input extremity of the pipe.
out_ : output extremity of the pipe.
external_trigger : Default=None, can be add through "add_external_trigger" instance

Methods
-------
add_external_trigger(link_instance): add an external trigger Link.
send(pickable) : sends a pickable object (or the boolean returned by the 
	condition).
recv(pickable) : receive a pickable object.
		"""
		
		self.out_,self.in_=Pipe()
		self.external_trigger=None
		self.condition=condition
	
	def add_external_trigger(self,link_instance):
		self.external_trigger=link_instance
	
	def send(self,value):
		if condition==None:
			self.out_.send(value)
		else:
			if self.external_trigger=None:
				self.out_.send(condition.evaluate(value))
			else:
				self.out_.send(condition.evaluate(value,self.external_trigger))
		
	
	def recv(self):
		return self.in_.recv()
	
	

class Condition:
	"""metaclass for all Links' conditions. Must implement the evaluate method"""
	__metaclass__= abc.ABCMeta

	@abc.abstractmethod
	def evaluate(self):
		pass