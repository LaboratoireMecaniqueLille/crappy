import time
import crappy

crappy.blocks.MasterBlock.instances = []  # Init masterblock instances

# t0=time.time()
try:
    # Creating objects
    # instronSensor=crappy.sensor.ComediSensor(device='/dev/comedi0',channels=[0,1],gain=[10,10])
    # sensor = crappy.sensor.ComediSensor(device='/dev/comedi1', channels=[0, 1, 2, 3],
    #                                     gain=[0.02, 100000, 0.01 * 2., 500])  # dist is multiplied by 2 to be correct
    # sensor = crappy.sensor.LabJackSensor(channels=[0],gain=[1],chan_range=10,mode="streamer",scanRate=10,scansPerRead=5)
    sensor = crappy.sensor.LabJackSensor(channels=[0], gain=1, resolution=12, chan_range=0.01, mode="single")  #
    # dist is multiplied by 2 to be correct
    # instronSensor=crappy.sensor.ComediSensor(device='/dev/comedi0',channels=[0,1],gain=[10,10000]) # 10 times the gain on the machine if you go through an usb dux sigma
    # cmd_traction=crappy.actuator.LabJackActuator(channel="TDAC2", gain=1, offset=0)
    # cmd_traction2=crappy.actuator.LabJackActuator(channel="TDAC3", gain=1, offset=0)
    # cmd_torsion=crappy.actuator.ComediActuator(device='/dev/comedi1', subdevice=1, channel=2, range_num=0, gain=1, offset=0)

    # Initialising the outputs

    # cmd_torsion.set_cmd(0)
    # cmd_traction.set_cmd(0)

    ########################################### Creating blocks
    # send_freq=400, actuator=cmd_traction, waveform=['sinus','sinus','sinus'], freq=[0.5,2,1], time_cycles=[10,10,10], amplitude=[1,2,4], offset=[0,0,0], phase=[0,0,0], repeat=True
    # send_freq=400, actuator=cmd_torsion, waveform=['sinus','triangle','sinus'], freq=[0.5,2,1], time_cycles=[10,10,10], amplitude=[0,0,0], offset=[0,0,0], phase=[np.pi,np.pi,np.pi], repeat=True
    # stream=crappy.blocks.MeasureByStep(instronSensor,labels=['t(s)','signal','signal2'],freq=200)
    # stream = crappy.blocks.MeasureByStep(sensor, labels=['t(s)', 'def(%)', 'F(N)', 'dist', 'C(Nm)'], freq=100)
    stream = crappy.blocks.MeasureByStep(sensor, labels=['t(s)', 'T'], freq=100)
    # stream=crappy.blocks.Streamer(sensor,labels=['t(s)','signal','signal2'])
    # stream=crappy.blocks.MeasureComediByStep(instronSensor, labels=['t(s)','V'], freq=1000.)
    # traction=crappy.blocks.SignalGenerator(path=[{"waveform":"sinus","time":100,"phase":0,"amplitude":1,"offset":0,"freq":2}],
    # send_freq=400,repeat=True)
    # torsion=crappy.blocks.SignalGenerator(path=[{"waveform":"triangle","time":50,"phase":0,"amplitude":5,"offset":-0.5,"freq":1}]
    # ,send_freq=400,repeat=False,labels=['t(s)','signal'])

    # send_output=crappy.blocks.CommandComedi([cmd_traction,cmd_traction2])
    compacter = crappy.blocks.Compacter(2)
    # compacter2=crappy.blocks.Compacter(400)
    # save=crappy.blocks.Saver("/home/ilyesse/Bureau/delete_me3.txt")
    graph = crappy.blocks.Grapher("static", ('t(s)', 'T'))
    # graph_stat=crappy.blocks.Grapher("dynamic",(0,2))
    # graph2=crappy.blocks.Grapher("dynamic",('t(s)','ang(deg)'),('t(s)','dep(mm)'))
    # graph3=crappy.blocks.Grapher("dynamic",(0,4))

    # Creating links
    # crappy.links.Filter(labels=['dist(deg)'],mode="median",size=50)
    # condition=[crappy.links.Filter(labels=['V'],mode="median",size=50),crappy.links.Filter(labels=['t(s)'],mode="mean",size=50)]
    link1 = crappy.links.Link()
    link2 = crappy.links.Link()
    link3 = crappy.links.Link()
    link4 = crappy.links.Link()
    link5 = crappy.links.Link()
    link6 = crappy.links.Link()
    link7 = crappy.links.Link()

    # Linking objects
    stream.add_output(link1)
    # traction.add_output(link2)
    # traction.add_output(link1)
    # torsion.add_output(link5)

    compacter.add_input(link1)
    # compacter2.add_input(link2)
    # send_output.add_input(link2)
    compacter.add_output(link3)
    # compacter.add_output(link4)
    # compacter.add_output(link5)
    # compacter.add_output(link6)
    # compacter.add_output(link7)

    # save.add_input(link5)

    graph.add_input(link3)

    # graph_stat.add_input(link5)
    # graph2.add_input(link4)
    # graph3.add_input(link7)
    # Starting objects
    t0 = time.time()
    for instance in crappy.blocks.MasterBlock.instances:
        instance.t0 = t0

    for instance in crappy.blocks.MasterBlock.instances:
        instance.start()

    # Waiting for execution

    # Stopping objects

    # for instance in crappy.blocks.MasterBlock.instances:
    # instance.stop()

except KeyboardInterrupt:
    for instance in crappy.blocks.MasterBlock.instances:
        instance.stop()
