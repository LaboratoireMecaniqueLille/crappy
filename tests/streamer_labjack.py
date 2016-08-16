import time
import crappy2 as crappy

crappy.blocks.MasterBlock.instances = []  # Init masterblock instances

stream = 1
try:
	graph = crappy.blocks.Grapher(('t(s)', 'T'), length=20)
	if stream:
		sensor = crappy.sensor.LabJackSensor(channels=["AIN0"], mode="streamer", scan_rate=100000)
		# sensor_args = {"channels": [0, 1], "mode": "thermocouple"}
		# labjack = crappy.technical.LabJack(channels=[0, 1], mode="single")
		stream = crappy.blocks.Streamer(sensor, labels=['t(s)', 'T'])

		# save = crappy.blocks.Saver("/home/francois/Code/Tests_Python/test_streaming.txt")
		#
		# Links
		Link_StreamToCompacter = crappy.links.Link()  # Link_StreamToCompacter = crappy.links.Link()  #
		stream.add_output(Link_StreamToCompacter)
		graph.add_input(Link_StreamToCompacter)

	# #
	else:
		sensor = crappy.sensor.LabJackSensor(channels=["AIN0"], mode="single")
		stream = crappy.blocks.MeasureByStep(sensor, labels=['t(s)', 'T'], freq=100)
		compacter = crappy.blocks.Compacter(200)
		Link_StreamToCompacter = crappy.links.Link()  # Link_StreamToCompacter = crappy.links.Link()  #
		stream.add_output(Link_StreamToCompacter)
		compacter.add_input(Link_StreamToCompacter)
		#
		Link_CompacterToGraph = crappy.links.Link()
		compacter.add_output(Link_CompacterToGraph)
		graph.add_input(Link_CompacterToGraph)
	# Starting objects
	t0 = time.time()

	for instance in crappy.blocks.MasterBlock.instances:
		instance.t0 = t0

	for instance in crappy.blocks.MasterBlock.instances:
		instance.start()

except KeyboardInterrupt:
	sensor.close()
	for instance in crappy.blocks.MasterBlock.instances:
		instance.stop()
