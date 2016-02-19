# coding: utf-8
from ._metaCondition import MetaCondition


class Bouncer(MetaCondition):
	def __init__(self,labels=[]):
		self.labels=labels

	def evaluate(self,value):
		return value[[label for label in labels]]