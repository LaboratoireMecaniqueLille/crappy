import crappy
import numpy as np
from time import sleep, time

#
opendaq = crappy.technical.OpenDAQ(channels=1, nsamples=100, gain=0)
# labjack = crappy.technical.LabJack(actuator={'channel': 'TDAC0'},
#                                    sensor={'channels': 'AIN0', 'resolution': 8, 'range_num': 10})
#
# measure_labjack = crappy.blocks.MeasureByStep(labjack, labels=['t(s)', 'AIN0'])
# compact_labjack = crappy.blocks.Compacter(100)
# grapher_labjack = crappy.blocks.Grapher(('t(s)', 'AIN0'), length=10)
# crappy.link(measure_labjack, compact_labjack, name='pou')
# crappy.link(compact_labjack, grapher_labjack, name='dav')

measure_opendaq = crappy.blocks.MeasureByStep(opendaq, labels=['t(s)', 'AN1'])
grapher_opendaq = crappy.blocks.Grapher(('t(s)', 'AN1'), length=100)
compact_opendaq = crappy.blocks.Compacter(100)
crappy.link(measure_opendaq, compact_opendaq, name='toto')
crappy.link(compact_opendaq, grapher_opendaq, name='foo')

crappy.start()

volt = 0
x = np.linspace(0, 2 * np.pi, 100)
sine = np.sin(x)
period = 0
# damping = 0.
labjack_actuator = 0.
t0 = 0
while True:
  try:
    if labjack_actuator:
      for i in xrange(len(sine)):lmsfkjkdlbjflkuj
        volt = sine[i] * damping
        labjack.set_cmd(volt)
        opendaq.set_cmd(volt)
        sleep(1)
        damping = damping + 0.001 if damping < 1 else 1
        pass
        period += 1
    else:
      volt = (sine * 2047 + 2048).round()
      for i in xrange(len(sine)):
        print 'valeur actuelle:', volt[i]
        opendaq.set_cmd(volt[i])

        sleep(0.0005)
      period += 1
      # print 'frequence:', time() - t0
  except Exception as e:
    print 'nombre de periodes:', period
    print e
    raise

  import time

  # from opendaq import DAQ
  # dq = DAQ("/dev/ttyUSB0")
  # liste = [1, 2, 3]
  # dq.conf_adc(pinput=1, ninput=0, gain=2, nsamples=20)
  # i = 0.
  # while True:
  #     try:
  #         x = i/1000 % 3
  #         data = dq.read_analog()
  #         print data
  #         i += 1
  #         # time.sleep(0.1)
  #     except:
  #         dq.close()
  #         break
  # # gain et offset : 1004, -12
