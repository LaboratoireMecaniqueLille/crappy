# coding: utf-8
from _meta import MasterBlock

class Reader(MasterBlock):
	"""
Children class of MasterBlock. Read and print the input Link.
	"""
	def __init__(self,k):
		"""
Create a reader that prints k and the input data in continuous.

Parameters
----------
k : printable
	Some identifier for this particular instance of Reader
		"""
		self.k=k  
		
	def main(self):
		while True:
			for input_ in self.inputs:
				self.data=input_.recv()
			print self.k,self.data
