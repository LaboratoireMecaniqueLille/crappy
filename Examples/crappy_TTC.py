# This file should be used to run monotonous testing on traction-torsion -compression machine.

import time
import numpy as np
import crappy2
import pandas as pd

crappy2.blocks.MasterBlock.instances = []  # Init masterblock instances


# class condition_signal(crappy2.links.MetaCondition):
#     def __init__(self, input_value_label):
#         self.input_value_label = input_value_label
#
#     def evaluate(self, value):
#         value['signal'] = value.pop(self.input_value_label)
#         return value


class EvalStrain(crappy2.links.MetaCondition):
    def __init__(self, k):
        self.surface = 110.74 * 10 ** (-6)
        self.I = np.pi * ((25 * 10 ** -3) ** 4 - (22 * 10 ** -3) ** 4) / 32
        self.rmoy = ((25 + 22) * 10 ** (-3)) / 2
        self.size = 20
        self.labels = ['dist(deg)', 'def(%)', 'C(Nm)', 'F(N)']
        self.FIFO = [[] for label in self.labels]
        self.k = k  # for testing

    def evaluate(self, value):
        # if self.k==1: # for testing
        # print value
        for i, label in enumerate(self.labels):
            # print self.FIFO[i]
            self.FIFO[i].insert(0, value[label])
            if len(self.FIFO[i]) > self.size:
                self.FIFO[i].pop()
            result = np.mean(self.FIFO[i])
            value[label] = result
        value['tau(Pa)'] = ((value['C(Nm)'] / self.I) * self.rmoy)
        value['sigma(Pa)'] = (value['F(N)'] / self.surface)
        value['eps_tot(%)'] = np.sqrt((value['def(%)']) ** 2 + ((value['dist(deg)']) ** 2) / 3.)
        # if self.k==1:
        # print value
        return value


if __name__ == '__main__':
    try:
        # Creating objects
        # we measure the offset to have the correct value for def and dist
        instronSensor = crappy2.sensor.ComediSensor(device='/dev/comedi0', channels=[0, 1, 2, 3],
                                                    gain=[0.01998, 99660, 0.0099856 * 2.,
                                                         499.5])  # dist is multiplied by 2 to be correct
        offset = np.array([0., 0., 0., 0.])
        for i in range(100):
            for j in range(0, 4, 2):
                offset[j] += instronSensor.get_data(j)[1] / 100.
        # offset-=np.array([0,1806,0,0.175])
        offset *= -1
        # end of the offset measure
        # 10 times the gain on the machine if you go through an usb dux sigma
        instronSensor = crappy2.sensor.ComediSensor(device='/dev/comedi0', channels=[0, 1, 2, 3],
                                                    gain=[0.01998, 99660, 0.0099856 * 2., 499.5],
                                                    offset=offset)

        # with comedi as output:
        # cmd_traction=crappy2.actuator.ComediActuator(device='/dev/comedi1', subdevice=1, channel=1, range_num=0,
        #  gain=8*100, offset=0)
        # cmd_torsion=crappy2.actuator.ComediActuator(device='/dev/comedi1', subdevice=1, channel=2, range_num=0,
        #  gain=8*100/2., offset=0) # divide dist by 2 for the testing machine

        # with Labjack as output:
        cmd_traction = crappy2.actuator.LabJackActuator(channel="TDAC2", gain=8 * 100, offset=0)
        cmd_torsion = crappy2.actuator.LabJackActuator(channel="TDAC3", gain=8 * 100 / 2., offset=0)

        # Initialising the outputs

        cmd_torsion.set_cmd(0)
        cmd_traction.set_cmd(0)
        time.sleep(0.5)
        cmd_torsion.set_cmd(0)
        cmd_traction.set_cmd(0)
        print "ready ?"  # Wait for you to start the test on the machine
        raw_input()

        # Creating blocks
        stream = crappy2.blocks.MeasureByStep(instronSensor, labels=['t(s)', 'def(%)', 'F(N)', 'dist(deg)', 'C(Nm)'])
        # stream=crappy2.blocks.StreamerComedi(instronSensor, labels=['t(s)','def(%)','F(N)','dist(deg)','C(Nm)'],
        #                                     freq=200.)

        # {"waveform":"detection","cycles":2}
        # {"waveform":"trefle","gain":0.0005,"cycles":1,"offset":[0.0002,-0.0002]},
        # {"waveform":"sablier","gain":0.0005,"cycles":1,"offset":[-0.0002,0.0002]},
        # {"waveform":"sablier","gain":0.0005,"cycles":1,"offset":[-0.0002,0.0002]},
        # {"waveform":"goto","position":[0,0]}
        multipath = crappy2.blocks.MultiPath(
            path=[{"waveform": "circle", "gain": 0.0006, "cycles": 200, "offset": [-0.0006, 0]},
                  {"waveform": "detection", "cycles": 1},
                  {"waveform": "circle", "gain": 0.0008, "cycles": 200, "offset": [-0.0006, 0]},
                  # {"waveform":"goto","mode":"plastic_def","target":0.002,"position":[-10,0]}
                  {"waveform": "detection", "cycles": 1}],
            send_freq=200, dmin=22, dmax=25, repeat=False, normal_speed=6.6 * 10 ** (-4))

        ttc = crappy2.blocks.CommandComedi([cmd_traction, cmd_torsion], signal_label=['def(%)', 'dist(deg)'])
        # torsion=crappy2.blocks.CommandComedi([cmd_torsion])

        compacter_data = crappy2.blocks.Compacter(200)
        save = crappy2.blocks.Saver("/home/corentin/Bureau/data_labjack_200_cycles.txt")
        graph_traction = crappy2.blocks.Grapher("static", ('sigma(Pa)', 'tau(Pa)'))
        graph_torsion = crappy2.blocks.Grapher("static", ('def(%)', 'dist(deg)'))
        # graph_torsion=crappy2.blocks.Grapher("static",('t(s)','def(%)'))

        compacter_path = crappy2.blocks.Compacter(200)
        save_path = crappy2.blocks.Saver("/home/corentin/Bureau/data_out_labjack_200_cycles.txt")
        graph_path = crappy2.blocks.Grapher("dynamic", ('t(s)', 'def(%)'))
        graph_path2 = crappy2.blocks.Grapher("dynamic", ('t(s)', 'dist(deg)'))
        # graph_torsion=crappy2.blocks.Grapher("dynamic",('t(s)','C(Nm)'))
        # graph_stat=crappy2.blocks.Grapher("dynamic",(0,2))
        # graph2=crappy2.blocks.Grapher("dynamic",(0,3))
        # graph3=crappy2.blocks.Grapher("dynamic",(0,4))

        # Creating links

        link1 = crappy2.links.Link(EvalStrain(k=1))
        link2 = crappy2.links.Link()
        link3 = crappy2.links.Link()
        link4 = crappy2.links.Link(EvalStrain(k=2))
        link5 = crappy2.links.Link()
        link6 = crappy2.links.Link()
        link7 = crappy2.links.Link()
        link8 = crappy2.links.Link()
        link9 = crappy2.links.Link()
        link10 = crappy2.links.Link()
        link11 = crappy2.links.Link()

        # Linking objects
        stream.add_output(link1)
        stream.add_output(link4)

        compacter_data.add_input(link1)
        compacter_data.add_output(link2)
        compacter_data.add_output(link3)
        compacter_data.add_output(link11)
        # compacter_data.add_output(link6)
        # compacter_data.add_output(link7)
        # compacter_data.add_output(link10)

        graph_traction.add_input(link2)
        graph_torsion.add_input(link3)
        save.add_input(link11)

        multipath.add_input(link4)
        multipath.add_output(link5)
        multipath.add_output(link8)
        # multipath.add_output(link9)

        compacter_path.add_input(link5)
        compacter_path.add_output(link6)
        compacter_path.add_output(link7)
        compacter_path.add_output(link10)

        save_path.add_input(link10)

        graph_path.add_input(link6)
        graph_path2.add_input(link7)

        ttc.add_input(link8)
        # torsion.add_input(link9)

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

    except (Exception, KeyboardInterrupt) as e:
        print "Exception in main :", e
        for instance in crappy2.blocks.MasterBlock.instances:
            try:
                instance.stop()
                print "instance stopped : ", instance
            except Exception as e:
                print e
