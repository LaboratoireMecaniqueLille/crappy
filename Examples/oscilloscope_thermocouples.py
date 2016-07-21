import time
import numpy as np
import crappy2 as crappy

crappy.blocks.MasterBlock.instances = []  # Init masterblock instances

try:

	sensor = crappy.sensor.LabJackSensor(channels=["AIN0", "AIN1", "AIN2", "AIN3"], mode="thermocouple")

	stream = crappy.blocks.MeasureByStep(sensor, labels=['t(s)', 'T', 'T2', 'T3', 'T4'], freq=800)

	compacter = crappy.blocks.Compacter(20)
	# save = crappy.blocks.Saver("/home/corentin/Code/crappy/tests_francois/Thermocouples_20160621/"
	#                            "thermocouples_instron_1Hz_30Nm.txt")

	graph = crappy.blocks.Grapher(('t(s)', 'T'), ('t(s)', 'T2'), ('t(s)', 'T3'), ('t(s)', 'T4'), length=180)  #
	# Links
	Link_StreamToCompacter = crappy.links.Link()  # Link_StreamToCompacter = crappy.links.Link()  #
	stream.add_output(Link_StreamToCompacter)
	compacter.add_input(Link_StreamToCompacter)
	#
	Link_CompacterToGraph = crappy.links.Link()
	compacter.add_output(Link_CompacterToGraph)
	graph.add_input(Link_CompacterToGraph)
	#
	# Link_CompacterToSave = crappy.links.Link()
	# compacter.add_output(Link_CompacterToSave)
	# save.add_input(Link_CompacterToSave)

	# Starting objects
	t0 = time.time()

	for instance in crappy.blocks.MasterBlock.instances:
		instance.t0 = t0

	for instance in crappy.blocks.MasterBlock.instances:
		instance.start()

except KeyboardInterrupt:
	for instance in crappy.blocks.MasterBlock.instances:
		instance.stop()
