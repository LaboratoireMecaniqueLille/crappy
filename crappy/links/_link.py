# coding: utf-8
from multiprocessing import Pipe
import copy
from functools import wraps
#import errno
import os
import signal

class TimeoutError(Exception):
    pass

def timeout_func(f):
	def _handle_timeout(signum, frame):
		raise TimeoutError("timeout error in pipe send")
	
	def wrapper(*args):
		signal.signal(signal.SIGALRM, _handle_timeout)
		signal.setitimer(signal.ITIMER_REAL,args[0].timeout)
		try:
			result = f(*args)
		finally:
			signal.alarm(0)
		return result
	return wrapper


class Link(object):
	"""
Main class for links. All links should inherit this class.
	"""
	def __init__(self,condition=None,timeout=0.1,action="warn"):
		"""
Link([condition=None])

Creates a pipe with a condition as attribute, and is used to transfer 
information between blocks using a pipe, triggered by the condition.

Parameters
----------
condition : Children class of links.Condition, optionnal
	Each "send" call will pass through the condition.evaluate method and sends
	the returned value.
	You can pass a list of conditions, the link will execute them in order.
	
Attributes
----------
in_ : input extremity of the pipe.
out_ : output extremity of the pipe.
external_trigger : Default=None, can be add through "add_external_trigger" instance

Methods
-------
add_external_trigger(link_instance): add an external trigger Link.
send : send the value, or a modified value if you pass it through a condition.
recv(blocking=True) : receive a pickable object. If blocking=False, return None
if there is no data
		"""
		
		self.in_,self.out_=Pipe(duplex=False)
		self.external_trigger=None
		self.condition=condition
		self.timeout=timeout
		self.action=action
	
	def add_external_trigger(self,link_instance):
		self.external_trigger=link_instance
		self.condition.external_trigger=link_instance
	
	def send(self,value):
		try:
			self.send_timeout(value)
		except TimeoutError as e:
			if self.action=="warn":
				print "WARNING : Timeout error in pipe send!"
			elif self.action=="kill":
				print "Killing Link : ", e
				raise
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in link : ", e
			raise
			
			
	@timeout_func
	def send_timeout(self,value):
		"""Send data through the condition.evaluate(value) function"""
		try:
			if self.condition==None:
				self.out_.send(value)
			else:
				try:
					for i in range(len(self.condition)):
						value=self.condition[i].evaluate(copy.copy(value))
				except TypeError: # if only one condition
					value=self.condition.evaluate(copy.copy(value))
				if not value is None:
					self.out_.send(value)
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in link : ", e
			raise
	
	def recv(self,blocking=True):
		"""Receive data. If blocking=False, return None if there is no data"""
		try:
			if blocking:
				return self.in_.recv()
			else:
				if self.in_.poll():
					return self.in_.recv()
				else:
				  return None
		except (Exception,KeyboardInterrupt) as e:
			print "Exception in link : ", e
			raise