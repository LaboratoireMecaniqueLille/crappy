# coding: utf-8
try:
	from ._clSerial import ClSerial
	from ._jaiSerial import JaiSerial
except:
	print "cannot import ClSerial"