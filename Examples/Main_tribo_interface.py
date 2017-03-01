#!/usr/bin/env python
import crappy
import numpy as np
import time
# import Tix
from Tkinter import *


# class conditionfiltree(crappy.links.Condition):
#   def __init__(self, labels=[], mode="mean", size=10):
#     self.mode = mode
#     self.size = size
#     self.labels = labels
#     self.FIFO = [[] for label in self.labels]
#     self.test = False
#     self.blocking = False

# def evaluate(self, value):
#   # print "1"
#   recv = self.external_trigger.recv(blocking=self.blocking)  # first run is blocking, others are not
#   self.blocking = False
#   if recv == 1:
#     # print 'EUREKA'
#     self.test = True
#
#   elif recv == 0:
#     # print 'je recois rien'
#     self.test = False

# for i, label in enumerate(self.labels):
#   # print self.FIFO[i]
#   self.FIFO[i].insert(0, value[label])
#   if len(self.FIFO[i]) > self.size:
#     self.FIFO[i].pop()
#   if self.mode == "median":
#     result = np.median(self.FIFO[i])
#   elif self.mode == "mean":
#     result = np.mean(self.FIFO[i])
#   value[label + "_filtered"] = result
#
# if self.test:
#   return value
# else:
#   return None

# def eval_offset(device, duration):
#   """
#   function used to evaluate offset (call it at the beginning).
#   """
#   timeout = time.time() + duration  # duration secs from now
#   print 'Measuring offset (%d sec), please wait...' % duration
#   offset_channels = [[] for i in xrange(device.nb_channels)]
#   offsets = []
#   while True:
#     measures = device.get_data()[1]
#     for i in xrange(len(offset_channels)):
#       offset_channels[i].append(measures)
#
#     if time.time() > timeout:
#       for i in xrange(len(offset_channels)):
#         offsets.append(-np.mean(offset_channels[i]))
#       print 'offsets:', offsets
#       break
#   return offsets

# Device definition
comediSensor = crappy.sensor.ComediSensor(channels=[0, 1, 2],
                                          gain=[20613, 4125, 500],
                                          offset=[0, 0, 0])
# offsets = eval_offset(comediSensor, 1)
# comediSensor = crappy.sensor.ComediSensor(channels=[0, 1, 2],
#                                           gain=[20613, 4125, -500],
#                                           offset=offsets)
#
# conditioners = [crappy.technical.Conditionner_5018(port='/dev/ttyS5'),
#                 crappy.technical.Conditionner_5018(port='/dev/ttyS6'),
#                 crappy.technical.Conditionner_5018(port='/dev/ttyS7')]

measure_comedi = crappy.blocks.MeasureByStep(comediSensor,
                                     labels=['t(s)', 'Effort_normal(N)', 'Vitesse(tr/min)', 'Couple(N.m)'])

# VariateurTribo = crappy.technical.VariateurTribo(port='/dev/ttyS4')

labjack = crappy.actuator.LabJackActuator(channel="TDAC0",
                                          gain=1. / 399.32,
                                          offset=-17.73 / 399.32)
labjack_trigger = crappy.actuator.LabJackActuator(channel="DAC0",
                                                  gain=1.,
                                                  offset=0)

labjack_hydrau1 = crappy.actuator.LabJackActuator(channel="FIO2",
                                                  gain=1.,
                                                  offset=0)  # SUR Y01
labjack_hydrau2 = crappy.actuator.LabJackActuator(channel="FIO3",
                                                  gain=1.,
                                                  offset=0)  # SUR Y02

labjack.set_cmd(0)
labjack.set_cmd_ram(0, 46002)  # sets the pid off
labjack.set_cmd_ram(0, 46000)  # sets the setpoint at 0 newton

# saver = crappy.blocks.Saver("/home/tribo/tests_francois_mars2017/toto.csv", stamp='date')

graph_effort = crappy.blocks.Grapher(('t(s)', 'Effort_normal(N)'), length=50)
graph_vitesse = crappy.blocks.Grapher(('t(s)', 'Vitesse(tr/min)'), length=50)
graph_couple = crappy.blocks.Grapher(('t(s)', 'Couple(N.m)'), length=50)
# root = Tix.Tk()
# interface = crappy.blocks.InterfaceTribo(root,
#                                          VariateurTribo,
#                                          labjack,
#                                          labjack_trigger,
#                                          conditioners,
#                                          labjack_hydrau1,
#                                          labjack_hydrau2)
# interface.root.protocol("WM_DELETE_WINDOW", interface.on_closing)

crappy.link(measure_comedi, graph_effort)
crappy.link(measure_comedi, graph_vitesse)
crappy.link(measure_comedi, graph_couple)
# crappy.link(measure_comedi, interface)
# links
crappy.start()

# print 'top1'
# interface.mainloop()

# try:
#   var = interface.getInfo()
#   root.destroy()
# # print var
# except Exception as e:
#   print "Error: ", e
#   sys.exit(0)

# except KeyboardInterrupt:
#   VariateurTribo.actuator.stop_motor()
#   labjack.set_cmd(0)
#   labjack.set_cmd_ram(-41, 46000)
#   labjack.set_cmd_ram(0, 46002)
#   time.sleep(1)
#   labjack.close()
#   time.sleep(0.1)
#   VariateurTribo.close()
#   for instance in crappy.blocks._masterblock.MasterBlock.instances:
#     instance.stop()
# except Exception as e:
#   print e
# finally:
#   time.sleep(0.1)
#   VariateurTribo.close()
#   print "Hasta la vista Baby"
