#!/usr/bin/env python
import crappy
import numpy as np
import time
import Tix
from Tkinter import *

crappy.blocks.MasterBlock.instances = []

class ConditionFiltree(crappy.links.Condition):
    """
    Used to filter received data (instead of update the window at every increment, we update it by the mean or
    median value of measures.
    """

    def __init__(self, labels=[], mode="mean", array_size=10):
        self.mode = mode
        self.array_size = array_size
        self.labels = labels
        self.FIFO = [[] for label in self.labels]
        self.test = False
        self.blocking = False

    def evaluate(self, value):
        """Used to what ? """
        recv = self.external_trigger.recv(blocking=self.blocking)  # first run is blocking, others are not
        self.blocking = False
        self.test = True if recv else False

        for i, label in enumerate(self.labels):
            self.FIFO[i].insert(0, value[label])
            if len(self.FIFO[i]) > self.array_size:
                self.FIFO[i].pop()
            if self.mode == "median":
                result = np.median(self.FIFO[i])
            elif self.mode == "mean":
                result = np.mean(self.FIFO[i])
            value[label + "_filtered"] = result
            return value if self.test else None


def eval_offset(device, duration):
    """Method to evaluate offset of comedi at initialization."""
    timeout = time.time() + duration  # duration secs from now
    print 'Measuring offset (%d sec), please be patient...' % duration
    offsets = [[] for chan in device.channels]
    while True:
        offsets = device.get_data('all')[1]
        if time.time() > timeout:
            break
    return [np.mean(offsets[chan]) for chan in xrange(len(device.channels))]


try:
    # Defining COMEDI: acquire force, velocity, torque
    comediSensor = crappy.sensor.ComediSensor(channels=[0, 1, 2], gain=[20613, 4125, -500], offset=[0, 0, 0])
    [F0, V0, C0] = eval_offset(comediSensor, 2)
    comediSensor = crappy.sensor.ComediSensor(channels=[0, 1, 2], gain=[20613, 4125, -500], offset=[-F0, -V0, C0])
    measurebystep_effort = crappy.blocks.MeasureByStep(comediSensor, labels=['t(s)', 'F(N)', 'Vitesse', 'Couple'],
                                                       freq=500)

    # Defining CONDITIONERS: acquire gauges on pad
    conditioners = [crappy.technical.Conditionner_5018(port=prt) for prt in ['/dev/ttyS5', '/dev/ttyS6', '/dev/ttyS7']]

    # Defining VARIATEUR_TRIBO :
    # is an actuator because motor position can be set, but also a sensor because can send back its position.
    VariateurTribo = crappy.technical.VariateurTribo(port='/dev/ttyS4')

    # Defining LABJACK : to set_cmd on PID. There is also the labjack_hydrau already set, although not used.
    labjack = crappy.actuator.LabJackActuator(channel="TDAC0", gain=1. / 399.32, offset=-17.73 / 399.32)
    labjack_hydrau = crappy.actuator.LabJackActuator(channel="DAC0", gain=1., offset=0)  # for future usages ?

    # Initialize PID:
    labjack.set_cmd(0)  # initialize what ?
    labjack.set_cmd_ram(0, 46002)  # sets the pid off
    labjack.set_cmd_ram(0, 46000)  # sets the setpoint at 0 newton

    # Other useful blocks
    saver = crappy.blocks.SaverTriggered("/home/tribo/save_dir/openlog.txt")  # TO BE REPLACED WITH THE SAVER !!
    compacter = crappy.blocks.Compacter(100)

    graph_force = crappy.blocks.Grapher(('t(s)', 'F(N)'), length=50)
    graph_vitesse = crappy.blocks.Grapher(('t(s)', 'Vitesse'), length=50)
    graph_couple = crappy.blocks.Grapher(('t(s)', 'Couple'), length=50)

    # Links
    link_effort_to_compacter = crappy.links.Link()
    link_to_graph_force = crappy.links.Link()
    link_to_graph_vitesse = crappy.links.Link()
    link_to_graph_couple = crappy.links.Link()

    link_effort_to_interface = crappy.links.Link(condition=ConditionFiltree())
    link_to_saver = crappy.links.Link(condition=ConditionFiltree())

    # Linkin (park)
    measurebystep_effort.add_output(link_effort_to_compacter)  # To send to graphs and saver
    # (triggered by the interface)

    compacter.add_input(link_effort_to_compacter)
    compacter.add_output(link_to_graph_force)
    compacter.add_output(link_to_graph_vitesse)
    compacter.add_output(link_to_graph_couple)
    compacter.add_output(link_to_saver)

    graph_force.add_input(link_to_graph_force)
    graph_vitesse.add_input(link_to_graph_vitesse)
    graph_couple.add_input(link_to_graph_couple)
    saver.add_input(link_to_saver)

    # Defining the window, and linking objects outside the window inside the window (combo window)
    root = Tix.Tk()  # To initialize the window
    interface = crappy.blocks.InterfaceTribo(root, VariateurTribo, labjack, labjack_hydrau,
                                             conditioners)
    interface.root.protocol("WM_DELETE_WINDOW", interface.on_closing)

    # Linking to the interface
    measurebystep_effort.add_output(link_effort_to_interface)  # For displaying?
    interface.add_input(link_effort_to_interface)

    # Linking from the interface
    trigger_save = crappy.links.Link()  # Boolean to start recording on hard_disk
    interface.add_output(trigger_save)
    link_effort_to_interface.add_external_trigger(trigger_save)

    linkRecordDataPath = crappy.links.Link()
    linkRecordData = crappy.links.Link()

    link_to_saver.add_external_trigger(linkRecordData)
    interface.add_output(linkRecordDataPath)
    interface.add_output(linkRecordData)
    saver.add_input(linkRecordDataPath)

    for instance in crappy.blocks.MasterBlock.instances:
        instance.t0 = t0

    for instance in crappy.blocks.MasterBlock.instances:
        instance.start()

    interface.mainloop()

except KeyboardInterrupt or Exception:
    VariateurTribo.actuator.stop_motor()
    labjack.set_cmd(0)
    labjack.set_cmd_ram(-41, 46000)
    labjack.set_cmd_ram(0, 46002)
    time.sleep(1)
    labjack.close()
    time.sleep(0.1)
    VariateurTribo.close()
    for instance in crappy.blocks.MasterBlock.instances:
        instance.stop()
except Exception as e:
    raise e
finally:
    time.sleep(0.1)
    VariateurTribo.close()
    print "Hasta la vista Baby"
