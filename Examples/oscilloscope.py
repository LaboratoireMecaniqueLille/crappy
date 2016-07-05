import time
import crappy2

crappy2.blocks.MasterBlock.instances = []  # Init masterblock instances

# t0=time.time()
try:
    # Creating objects
    # instronSensor=crappy2.sensor.ComediSensor(device='/dev/comedi0',channels=[0,1],gain=[10,10])
    sensor = crappy2.sensor.ComediSensor(device='/dev/comedi0', channels=[0, 1, 2, 3],
                                        gain=[1, 1, 1, 1])  # dist is multiplied by 2 to be correct
    # sensor = crappy2.sensor.LabJackSensor(channels=[0], gain=[1], chan_range=10, mode="streamer", scanRate=10,
    #                                       scansPerRead=5)
    # sensor = crappy2.sensor.LabJackSensor(channels=[0], gain=1, resolution=12, chan_range=10, mode="single")  #
    # dist is multiplied by 2 to be correct
    # 10 times the gain on the machine if you go through an usb dux sigma
    # instronSensor = crappy2.sensor.ComediSensor(device='/dev/comedi0',channels=[0,1],gain=[10,10000])
    # cmd_traction=crappy2.actuator.LabJackActuator(channel="TDAC2", gain=1, offset=0)
    # cmd_traction2=crappy2.actuator.LabJackActuator(channel="TDAC3", gain=1, offset=0)
    # cmd_torsion = crappy2.actuator.ComediActuator(device='/dev/comedi1', subdevice=1, channel=2, range_num=0, gain=1,
    #                                               offset=0)

    # Initialising the outputs

    # cmd_torsion.set_cmd(0)
    # cmd_traction.set_cmd(0)

    # Creating blocks
    # send_freq=400, actuator=cmd_traction, waveform=['sinus','sinus','sinus'], freq=[0.5,2,1], time_cycles=[10,10,10], amplitude=[1,2,4], offset=[0,0,0], phase=[0,0,0], repeat=True
    # send_freq=400, actuator=cmd_torsion, waveform=['sinus','triangle','sinus'], freq=[0.5,2,1], time_cycles=[10,10,10], amplitude=[0,0,0], offset=[0,0,0], phase=[np.pi,np.pi,np.pi], repeat=True
    # stream=crappy2.blocks.MeasureByStep(instronSensor,labels=['t(s)','signal','signal2'],freq=200)
    # stream = crappy2.blocks.MeasureByStep(sensor, labels=['t(s)', 'def(%)', 'F(N)', 'dist', 'C(Nm)'], freq=100)
    stream = crappy2.blocks.MeasureByStep(sensor, labels=['t(s)', 'AIN1', 'AIN2', 'AIN3', 'AIN4'], freq=100)
    # stream=crappy2.blocks.Streamer(sensor,labels=['t(s)','signal','signal2'])
    # stream=crappy2.blocks.MeasureComediByStep(instronSensor, labels=['t(s)','V'], freq=1000.)
    # traction=crappy2.blocks.SignalGenerator(path=[{"waveform":"sinus","time":100,"phase":0,"amplitude":1,"offset":0,"freq":2}],
    # send_freq=400,repeat=True)
    # torsion=crappy2.blocks.SignalGenerator(path=[{"waveform":"triangle","time":50,"phase":0,"amplitude":5,"offset":-0.5,"freq":1}]
    # ,send_freq=400,repeat=False,labels=['t(s)','signal'])

    # send_output=crappy2.blocks.CommandComedi([cmd_traction,cmd_traction2])
    compacter = crappy2.blocks.Compacter(20)
    # compacter2=crappy2.blocks.Compacter(400)
    # save=crappy2.blocks.Saver("/home/ilyesse/Bureau/delete_me3.txt")
    graph = crappy2.blocks.Grapher("dynamic", ('t(s)', 'AIN1'))
    graph2 = crappy2.blocks.Grapher("dynamic", ('t(s)', 'AIN2'), ('t(s)', 'AIN3'), ('t(s)', 'AIN4'))
    # graph_stat=crappy2.blocks.Grapher("dynamic",(0,2))
    # graph2=crappy2.blocks.Grapher("dynamic",('t(s)','ang(deg)'),('t(s)','dep(mm)'))
    # graph3=crappy2.blocks.Grapher("dynamic",(0,4))

    # Creating links
    # crappy2.links.Filter(labels=['dist(deg)'],mode="median",size=50)
    # condition=[crappy2.links.Filter(labels=['V'],mode="median",size=50),crappy2.links.Filter(labels=['t(s)'],mode="mean",size=50)]
    link1 = crappy2.links.Link(name="link1")
    link2 = crappy2.links.Link(name="link2")
    link3 = crappy2.links.Link(name="link3")
    link4 = crappy2.links.Link()
    # link5 = crappy2.links.Link()
    # link6 = crappy2.links.Link()
    # link7 = crappy2.links.Link()

    # Linking objects
    stream.add_output(link1)
    # traction.add_output(link2)
    # traction.add_output(link1)
    # torsion.add_output(link5)

    compacter.add_input(link1)
    # compacter2.add_input(link2)
    # send_output.add_input(link2)
    compacter.add_output(link3)
    compacter.add_output(link4)
    # compacter.add_output(link5)
    # compacter.add_output(link6)
    # compacter.add_output(link7)

    # save.add_input(link5)

    graph.add_input(link3)

    # graph_stat.add_input(link5)
    graph2.add_input(link4)
    # graph3.add_input(link7)
    # Starting objects
    t0 = time.time()
    for instance in crappy2.blocks.MasterBlock.instances:
        instance.t0 = t0

    for instance in crappy2.blocks.MasterBlock.instances:
        instance.start()

        # Waiting for execution

        # Stopping objects

        # for instance in crappy2.blocks.MasterBlock.instances:
        # instance.stop()

except KeyboardInterrupt:
    for instance in crappy2.blocks.MasterBlock.instances:
        instance.stop()
