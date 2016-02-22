# coding: utf-8
import abc	

class MetaCondition:
	"""Metaclass for all Links conditions. Must implement the evaluate method."""
	__metaclass__= abc.ABCMeta

	@abc.abstractmethod
	def evaluate(self):
		"""This method is called by the Links and must always be implemented."""
		pass