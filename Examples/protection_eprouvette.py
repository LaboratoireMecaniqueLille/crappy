import time

import crappy

crappy.blocks.MasterBlock.instances = []  # Init masterblock instances

# t0=time.time()
if __name__ == '__main__':
    try:
        # Creating objects
        # Init : Offset determination (gain is kept as usual)
        instronSensor = crappy.sensor.ComediSensor(channels=[1, 3], gain=[-3749.3, -3198.9 * 1.18], offset=[0, 0])
        t, F2 = instronSensor.get_data(0)
        t, F4 = instronSensor.get_data(1)
        instronSensor = crappy.sensor.ComediSensor(channels=[1, 3], gain=[-3749.3, -3198.9 * 1.18], offset=[-F2, -F4])
        biaxeTech1 = crappy.technical.Biaxe(port='/dev/ttyS4')
        biaxeTech2 = crappy.technical.Biaxe(port='/dev/ttyS5')
        biaxeTech3 = crappy.technical.Biaxe(port='/dev/ttyS6')
        biaxeTech4 = crappy.technical.Biaxe(port='/dev/ttyS7')

        axes = [biaxeTech1, biaxeTech2, biaxeTech3, biaxeTech4]

        # Creating blocks
        # Declare the compacter that allows to transfer by block of a given length
        compacter_effort = crappy.blocks.Compacter(100)
        # save_effort=crappy.blocks.Saver("/home/biaxe/Bureau/Annie/effort.txt")
        # Init : Declaration of The graph
        graph_effort = crappy.blocks.Grapher(('t(s)', 'F2(N)'), ('t(s)', 'F4(N)'))

        # compacter_extenso=crappy.blocks.Compacter(150)
        # save_extenso=crappy.blocks.Saver("/home/biaxe/Bureau/Annie/extenso.txt")
        # graph_extenso=crappy.blocks.Grapher("dynamic",('t(s)','Exx(%)'),('t(s)','Eyy(%)'))

        effort = crappy.blocks.MeasureComediByStep(instronSensor, labels=['t(s)', 'F2(N)', 'F4(N)'], freq=100)
        # extenso=crappy.blocks.VideoExtenso(camera="Ximea",xoffset=0,yoffset=0,width=2048,height=2048,white_spot=True,display=True)

        # signalGenerator=crappy.blocks.SignalGenerator(path=[{"waveform":"hold","time":3},
        # {"waveform":"limit","cycles":3,"phase":0,"lower_limit":[50,'F(N)'],"upper_limit":[5,'Exx(%)']}],
        # send_freq=400,repeat=True,labels=['t(s)','signal'])

        # signalGenerator=crappy.blocks.SignalGenerator(path=[{"waveform":"limit","gain":1,"cycles":0.5,"phase":0,"lower_limit":[50,'F2(N)'],"upper_limit":[10,'Exx(%)']},
        # {"waveform":"limit","gain":0,"cycles":0.5,"phase":0,"lower_limit":[50,'F4(N)'],"upper_limit":[9.7,'Eyy(%)']},
        # {"waveform":"limit","gain":1,"cycles":0.5,"phase":-np.pi,"lower_limit":[50,'F2(N)'],"upper_limit":[10,'Exx(%)']}],
        # send_freq=400,repeat=True,labels=['t(s)','signal'])
        signalGenerator = crappy.blocks.SignalGenerator(
            path=[{"waveform": "protection", "gain": 1, "lower_limit": [-1, 'F2(N)'], "upper_limit": [10, 'F2(N)']}],
            send_freq=100, repeat=True)

        signalGenerator_horizontal = crappy.blocks.SignalGenerator(
            path=[{"waveform": "protection", "gain": 1, "lower_limit": [-1, 'F4(N)'], "upper_limit": [10, 'F4(N)']}],
            send_freq=100, repeat=True)

        biotens = crappy.blocks.CommandBiaxe(biaxe_technicals=[biaxeTech1, biaxeTech2], speed=-5000)  # vertical
        biotens_horizontal = crappy.blocks.CommandBiaxe(biaxe_technicals=[biaxeTech3, biaxeTech4], speed=-5000)
        # horizontal # speed must be <0 for traction

        # Creating links

        link1 = crappy.links.Link()
        link2 = crappy.links.Link()
        link12 = crappy.links.Link()
        link22 = crappy.links.Link()
        link3 = crappy.links.Link()
        link4 = crappy.links.Link()
        link5 = crappy.links.Link()
        link6 = crappy.links.Link()
        link7 = crappy.links.Link()
        link8 = crappy.links.Link()
        link9 = crappy.links.Link()
        link92 = crappy.links.Link()

        # Linking objects

        effort.add_output(link1)
        effort.add_output(link12)
        effort.add_output(link6)

        # extenso.add_output(link2)
        # extenso.add_output(link22)
        # extenso.add_output(link3)

        signalGenerator.add_input(link1)
        # signalGenerator.add_input(link2)
        signalGenerator.add_output(link9)

        signalGenerator_horizontal.add_input(link12)
        # signalGenerator_horizontal.add_input(link22)/media/biaxe/SSD1To/EssaiBiAxe2/cam_1/
        signalGenerator_horizontal.add_output(link92)

        biotens.add_input(link9)
        biotens_horizontal.add_input(link92)

        compacter_effort.add_input(link6)
        # compacter_effort.add_output(link7)
        compacter_effort.add_output(link8)

        # save_effort.add_input(link7)

        graph_effort.add_input(link8)

        # compacter_extenso.add_input(link3)
        # compacter_extenso.add_output(link4)
        # compacter_extenso.add_output(link5)

        # save_extenso.add_input(link4)

        # graph_extenso.add_input(link5)

        # Starting objects
        # print "top :",crappy.blocks._masterblock.MasterBlock.instances
        t0 = time.time()
        for instance in crappy.blocks.MasterBlock.instances:
            instance.t0 = t0

        for instance in crappy.blocks.MasterBlock.instances:
            instance.start()

    # Waiting for execution

    # Stopping objects

    except (Exception, KeyboardInterrupt) as e:
        print "Exception in main :", e
        for instance in crappy.blocks.MasterBlock.instances:
            try:
                instance.stop()
            except Exception as e:
                print e
        for axe in axes:
            try:
                axe.close()
            except Exception as e:
                print e

            # for manual setting :
            # biaxeTech3.actuator.set_speed(500);biaxeTech4.actuator.set_speed(500);biaxeTech1.actuator.set_speed(500);biaxeTech2.actuator.set_speed(500)
