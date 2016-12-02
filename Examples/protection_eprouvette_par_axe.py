import time
import crappy

crappy.blocks.MasterBlock.instances = []  # Init masterblock instances

# t0=time.time()
if __name__ == '__main__':
    try:
        # Creating objects
        # Init : Offset determination (gain is kept as usual)
        instronSensor = crappy.sensor.ComediSensor(channels=[1, 2, 3, 4],
                                                    gain=[-3749.3, -3749.3, -3198.9 * 1.18, -3749.3],
                                                    offset=[0, 0, 0, 0])
        t, F1 = instronSensor.get_data(0)
        t, F2 = instronSensor.get_data(1)
        t, F3 = instronSensor.get_data(2)
        t, F4 = instronSensor.get_data(3)
        instronSensor = crappy.sensor.ComediSensor(channels=[1, 2, 3, 4],
                                                    gain=[-3749.3, -3749.3, -3198.9 * 1.18, -3749.3],
                                                    offset=[-F1, -F2, -F3, -F4])
        biaxeTech1 = crappy.technical.Biaxe(port='/dev/ttyS4')
        biaxeTech2 = crappy.technical.Biaxe(port='/dev/ttyS5')
        biaxeTech3 = crappy.technical.Biaxe(port='/dev/ttyS6')
        biaxeTech4 = crappy.technical.Biaxe(port='/dev/ttyS7')

        axes = [biaxeTech1, biaxeTech2, biaxeTech3, biaxeTech4]

        # Creating blocks

        compacter_effort = crappy.blocks.Compacter(100)

        graph_effort = crappy.blocks.Grapher(('t(s)', 'F2(N)'), ('t(s)', 'F4(N)'), ('t(s)', 'F3(N)'),
                                              ('t(s)', 'F1(N)'))

        effort = crappy.blocks.MeasureComediByStep(instronSensor, labels=['t(s)', 'F1(N)', 'F2(N)', 'F3(N)', 'F4(N)'],
                                                    freq=100)

        signalGenerator1 = crappy.blocks.SignalGenerator(
            path=[{"waveform": "protection", "gain": 1, "lower_limit": [-10, 'F1(N)'], "upper_limit": [10, 'F1(N)']}],
            send_freq=100, repeat=True)

        signalGenerator2 = crappy.blocks.SignalGenerator(
            path=[{"waveform": "protection", "gain": 1, "lower_limit": [-10, 'F2(N)'], "upper_limit": [10, 'F2(N)']}],
            send_freq=100, repeat=True)
        signalGenerator3 = crappy.blocks.SignalGenerator(
            path=[{"waveform": "protection", "gain": 1, "lower_limit": [-10, 'F3(N)'], "upper_limit": [10, 'F3(N)']}],
            send_freq=100, repeat=True)

        signalGenerator4 = crappy.blocks.SignalGenerator(
            path=[{"waveform": "protection", "gain": 1, "lower_limit": [-10, 'F4(N)'], "upper_limit": [10, 'F4(N)']}],
            send_freq=100, repeat=True)

        biotens1 = crappy.blocks.CommandBiaxe(biaxe_technicals=[biaxeTech1], speed=-5000)  # vertical
        biotens2 = crappy.blocks.CommandBiaxe(biaxe_technicals=[biaxeTech2], speed=-5000)  # vertical
        biotens3 = crappy.blocks.CommandBiaxe(biaxe_technicals=[biaxeTech3], speed=-5000)  # vertical
        biotens4 = crappy.blocks.CommandBiaxe(biaxe_technicals=[biaxeTech4], speed=-5000)
        # horizontal # speed must be <0 for traction

        # Creating links

        link1 = crappy.links.Link()
        link2 = crappy.links.Link()
        link3 = crappy.links.Link()
        link4 = crappy.links.Link()
        link5 = crappy.links.Link()
        link6 = crappy.links.Link()

        link11 = crappy.links.Link()
        link21 = crappy.links.Link()
        link31 = crappy.links.Link()
        link41 = crappy.links.Link()

        # Linking objects

        effort.add_output(link1)
        effort.add_output(link2)
        effort.add_output(link3)
        effort.add_output(link4)

        effort.add_output(link5)

        signalGenerator1.add_input(link1)
        signalGenerator1.add_output(link11)

        signalGenerator2.add_input(link2)
        signalGenerator2.add_output(link21)

        signalGenerator3.add_input(link3)
        signalGenerator3.add_output(link31)

        signalGenerator4.add_input(link4)
        signalGenerator4.add_output(link41)

        biotens1.add_input(link11)
        biotens2.add_input(link21)
        biotens3.add_input(link31)
        biotens4.add_input(link41)

        compacter_effort.add_input(link5)
        compacter_effort.add_output(link6)

        graph_effort.add_input(link6)

        # Starting objects
        # print "top :",crappy.blocks._meta.MasterBlock.instances
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
