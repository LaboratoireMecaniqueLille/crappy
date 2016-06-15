import time
#import matplotlib
#matplotlib.use('Agg')
import crappy 
import pandas as pd
crappy.blocks._meta.MasterBlock.instances=[] # Init masterblock instances

if __name__ == '__main__':
	agilentSensor=crappy.sensor.Agilent34420ASensor(device='/dev/ttyUSB0',baudrate=9600,timeout=1)

	tension=crappy.blocks.MeasureAgilent34420A(agilentSensor,labels=['t(s)','tension(V)'],freq=1)
	compacter_tension=crappy.blocks.Compacter(4)
	save_tension=crappy.blocks.Saver("/home/essais-2015-3/Bureau/tension_coeff.txt")
	graph_tension=crappy.blocks.Grapher("dynamic",('t(s)','tension(V)')) #,('t(s)','tension(V)')

	link10=crappy.links.Link()
	link11=crappy.links.Link()
	link12=crappy.links.Link()


	tension.add_output(link10)
	compacter_tension.add_input(link10)
	compacter_tension.add_output(link11)
	compacter_tension.add_output(link12)
	graph_tension.add_input(link11)
	save_tension.add_input(link12)

	try:
		t0=time.time()
		for instance in crappy.blocks._meta.MasterBlock.instances:
			instance.t0(t0)

		for instance in crappy.blocks._meta.MasterBlock.instances:
			instance.start()

	########################################### Waiting for execution
		#time.sleep(10)
		#sum1 = summary.summarize(muppy.get_objects())
		#summary.print_(sum1)
		#while True:
			#time.sleep(10)
			#sum2 = summary.summarize(muppy.get_objects())
			#diff = summary.get_diff(sum1, sum2)
			#summary.print_(diff)
	########################################### Stopping objects

	except (Exception,KeyboardInterrupt) as e:
		print "Exception in main :", e
		#for instance in crappy.blocks._meta.MasterBlock.instances:
			#instance.join()
		for instance in crappy.blocks._meta.MasterBlock.instances:
			try:
				instance.stop()
				print "instance stopped : ", instance
			except:
				pass