import crappy
import numpy as np
from time import sleep, time

#
opendaq = crappy.technical.OpenDAQ(channels=[1, 2, 5, 6], nsamples=20, gain=0)
# labjack = crappy.technical.LabJack(actuator={'channel': 'TDAC0'},
#                                    sensor={'channels': 'AIN0', 'resolution': 8, 'range_num': 10})
#
# measure_labjack = crappy.blocks.MeasureByStep(labjack, labels=['t(s)', 'AIN0'])
# compact_labjack = crappy.blocks.Compacter(100)
# grapher_labjack = crappy.blocks.Grapher(('t(s)', 'AIN0'), length=10)
# crappy.link(measure_labjack, compact_labjack, name='pou')
# crappy.link(compact_labjack, grapher_labjack, name='dav')
labels = ['t(s)', 'AN1', 'AN2', 'AN5', 'AN6']
measure_opendaq = crappy.blocks.MeasureByStep(opendaq, labels=labels)
grapher_opendaq = crappy.blocks.Grapher([('t(s)', label) for label in labels[1:]], length=100)
saver_opendaq = crappy.blocks.Saver('/home/francois/A_Projects/015_test_opendaq/test.csv', stamp='rioye')
compact_opendaq = crappy.blocks.Compacter(100)
crappy.link(measure_opendaq, compact_opendaq, name='toto')
crappy.link(compact_opendaq, grapher_opendaq, name='foo')
crappy.link(compact_opendaq, saver_opendaq)
#
crappy.start()
#
# volt = 0
# x = np.linspace(0, 2 * np.pi, 100)
# sine = np.sin(x)
# period = 0
# # damping = 0.
# labjack_actuator = 0.
# t0 = 0
# while True:
#   try:
#     if labjack_actuator:
#       for i in xrange(len(sine)):lmsfkjkdlbjflkuj
#         volt = sine[i] * damping
#         labjack.set_cmd(volt)
#         opendaq.set_cmd(volt)
#         sleep(1)
#         damping = damping + 0.001 if damping < 1 else 1
#         pass
#         period += 1
#     else:
#       volt = (sine * 2047 + 2048).round()
#       for i in xrange(len(sine)):
#         print 'valeur actuelle:', volt[i]
#         opendaq.set_cmd(volt[i])
#
#         sleep(0.0005)
#       period += 1
#       # print 'frequence:', time() - t0
#   except Exception as e:
#     print 'nombre de periodes:', period
#     print e
#     raise

# import time
#
# from opendaq import DAQ
# dq = DAQ("/dev/ttyUSB0")
# dq.conf_adc(pinput=1, ninput=0, gain=2, nsamples=1)
# i = 0.
# t0 = time.time()
# while True:
#   try:
#     data = dq.read_all(nsamples=20)
#     i += 1
#     if time.time() - t0 > 10:
#       tfinal = time.time()
#       print 'frequence:', i / (tfinal - t0)
#       break
#   except:
#     dq.close()
#     break
