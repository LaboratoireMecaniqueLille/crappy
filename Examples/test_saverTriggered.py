#!/usr/bin/env python
import crappy2 as crappy 
import numpy as np
import time
#from interface import *
import Tix
from Tkinter import *
import tkFont
import tkFileDialog


crappy.blocks.MasterBlock.instances=[]


t0=time.time()

try:
  
	sensor=crappy.sensor.ComediSensor(channels=[0],gain=[20613],offset=[0])
	stream = crappy.blocks.MeasureByStep(sensor, labels=['t(s)', 'AIN0'], freq=100)
	compacter = crappy.blocks.Compacter(20)
	graph = crappy.blocks.Grapher(('t(s)', 'AIN0'))
	save=crappy.blocks.SaverTriggered("/home/tribo/save_dir/openlog.txt")
	
	
	
	link1 = crappy.links.Link(name="link1")
	link2 = crappy.links.Link(name="link2")	
	link3 = crappy.links.Link(name="link3")	
	linkPath = crappy.links.Link(name="linkPath")	
	stream.add_output(link1)
	
	compacter.add_input(link1)
	compacter.add_output(link2)
	compacter.add_output(link3)
	
	graph.add_input(link3)
	save.add_input(link2)
	save.add_input(linkPath)
	
	
	
	t0 = time.time()
	for instance in crappy.blocks.MasterBlock.instances:
	    instance.t0 = t0

	for instance in crappy.blocks.MasterBlock.instances:
	    instance.start()
	
	root = Tix.Tk()
	interface = crappy.blocks.InterfaceSendPath(root)
	interface.add_output(linkPath)
	interface.mainloop()
	
	
	try:
		var = interface.getInfo()
		root.destroy()
		#print var
	except Exception as e:
		print "Error: ", e
		sys.exit(0)
	
except KeyboardInterrupt:
	for instance in crappy.blocks.MasterBlock.instances:
		instance.stop()