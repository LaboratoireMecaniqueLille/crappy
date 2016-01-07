import time
#import matplotlib
#matplotlib.use('Agg')
import crappy 
import pandas as pd
crappy.blocks._meta.MasterBlock.instances=[] # Init masterblock instances


agilentSensor=crappy.sensor.Agilent34420ASensor(device='/dev/ttyUSB0',baudrate=9600,timeout=1)
instronSensor=crappy.sensor.ComediSensor(device='/dev/comedi0',channels=[0,1],gain=[10,10000],offset=[0,0])

tension=crappy.blocks.MeasureAgilent34420A(agilentSensor,labels=['t(s)','tension(V)'],freq=2)
compacter_tension=crappy.blocks.Compacter(4)
save_tension=crappy.blocks.Saver("/home/essais-2015-3/Bureau/tension_coeff.txt")
graph_tension=crappy.blocks.Grapher("dynamic",('t(s)','tension(V)')) #,('t(s)','tension(V)')

effort=crappy.blocks.MeasureComediByStep(instronSensor,labels=['t(s)','dep(mm)','F(N)'],freq=200)
compacter_effort=crappy.blocks.Compacter(200)
graph_effort=crappy.blocks.Grapher("dynamic",('t(s)','F(N)'))
save_effort=crappy.blocks.Saver("/home/essais-2015-3/Bureau/t_dep_F.txt")





link1=crappy.links.Link()
link3=crappy.links.Link()
link2=crappy.links.Link()
link10=crappy.links.Link()
link11=crappy.links.Link()
link12=crappy.links.Link()


tension.add_output(link10)
compacter_tension.add_input(link10)
compacter_tension.add_output(link11)
compacter_tension.add_output(link12)
graph_tension.add_input(link11)
save_tension.add_input(link12)

effort.add_output(link1)
compacter_effort.add_input(link1)
compacter_effort.add_output(link2)
compacter_effort.add_output(link3)

graph_effort.add_input(link2)
save_effort.add_input(link3)

try:
	t0=time.time()
	for instance in crappy.blocks._meta.MasterBlock.instances:
		instance.set_t0(t0)

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