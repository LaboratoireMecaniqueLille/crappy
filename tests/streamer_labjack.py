from time import time, sleep
from crappy2.blocks import MasterBlock, Grapher, Saver, MeasureByStep, Streamer, Compacter
from crappy2.sensor import LabJackSensor
from crappy2.links import Link
import random
from crappy2.technical import DataPicker
import crappy2

import signal
import os

def loop1(scan_rate, scan_per_read):
    MasterBlock.instances = []  # Init masterblock instances

    stream = 0
    try:
        graph = Grapher(('t(s)', 'AIN0'), ('t(s)', 'AIN1'), ('t(s)', 'AIN2'), length=20)
        if stream:
            sensor = LabJackSensor(mode="streamer", scan_rate_per_channel=scan_rate, scans_per_read=scan_per_read,
                                   channels=[0, 1, 2])
            stream = Streamer(sensor=sensor)
            compacter = Compacter(50)

            Link_to_DataPicker = Link()
            Link_to_compacter = Link()

            stream.add_output(Link_to_DataPicker)
            # # save = DataPicker(Link_to_DataPicker)
            # compacter.add_input(Link_to_DataPicker)
            #
            # compacter.add_output(Link_to_compacter)

            graph.add_input(Link_to_DataPicker)

        else:
            sensor = LabJackSensor(mode="single", channels=['AIN0', 'AIN1', 'AIN2'])
            stream = MeasureByStep(sensor=sensor, labels=['t(s)', 'AIN0', 'AIN1', 'AIN2'])

            Link_to_MeasureByStep = Link()
            Link_to_Grapher = Link()
            stream.add_output(Link_to_MeasureByStep)
            compacter = Compacter(50)

            compacter.add_input(Link_to_MeasureByStep)
            compacter.add_output(Link_to_Grapher)
            graph.add_input(Link_to_Grapher)

        t0 = time()

        for instance in MasterBlock.instances:
            instance.t0 = t0

        for instance in MasterBlock.instances:
            instance.start()

    except KeyboardInterrupt:
        sensor.close()

        for instance in MasterBlock.instances:
            instance.stop()

rate1 = 33333
rate2 = int(rate1 / 10.)
loop1(rate1, rate2)

