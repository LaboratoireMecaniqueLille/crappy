from multiprocessing import Pipe,Process
import time
import copy
import Blocks
import Links


#def cond_1(value):
	
	#if value[1]%10==0:
		#return True
	#else:
		#return False

########################################### Creating objects
dispatch=Blocks.Dispatcher()
stream=Blocks.Streamer()
#read=Blocks.Reader(1)
#read2=Blocks.Reader(2)
compact=Blocks.Compacter(500)
save=Blocks.Saver("/home/corentin/Bureau/test_save.txt")
graph=Blocks.Grapher("static",(0,1),(0,1),(0,1),(0,1))
graph2=Blocks.Grapher("dynamic",(0,1),(0,1),(0,1),(0,1))
########################################### Creating links
link3=Links.Link()

link2=Links.Link()
link1=Links.Link()
link4=Links.Link()
link5=Links.Link()
########################################### Linking objects
stream.add_output(link1)
#print "top"
dispatch.add_input(link1)
#dispatch.add_output(link2)
#print "top2"
dispatch.add_output(link3)
#read.add_input(link2)
compact.add_input(link3)
compact.add_output(link4)
save.add_input(link4)
graph.add_input(link2)
graph2.add_input(link5)
compact.add_output(link2)
compact.add_output(link5)
########################################### Starting objects

for instance in Blocks.MasterBlock.instances:
    instance.start()

########################################### Waiting for execution
time.sleep(1)
time.sleep(50)

########################################### Stopping objects

for instance in Blocks.MasterBlock.instances:
    instance.stop()
