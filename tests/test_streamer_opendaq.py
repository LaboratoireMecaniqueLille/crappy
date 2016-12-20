# import timeit
#
# from opendaq import DAQ
# import time
#
# dq = DAQ('/dev/ttyUSB1')
# stream_exp = dq.create_stream(mode=0,
#                               # 0:ANALOG_INPUT 1:ANALOG_OUTPUT 2:DIGITAL_INPUT 3:DIGITAL_OUTPUT 4:COUNTER_INPUT 5:CAPTURE_INPUT
#                               period=1,
#                               # 0:65536
#                               npoints=0,
#                               # 0:65536
#                               continuous=True,
#                               buffersize=100)
# stream_exp.analog_setup(pinput=2, ninput=5, gain=0, nsamples=0)
#
# # stream_exp2 = dq.create_stream(mode=0, period=1, npoints=0, continuous=True)
# # stream_exp2.analog_setup(pinput=4, gain=0, nsamples=254)
# dq.start()
#
# def get_stream():
#   filling = []
#   lengths = []
#   nb = 0
#   while True:
#     # print 'filling au debut:', len(filling)
#     data = stream_exp.read()
#     filling.extend(data)
#     yield filling[:100]
#     del filling[:100]
#     nb += 100
#     print 'nombre acquisition:', nb
#     print 'donnes dans filling:', len(filling)
#     # dq.flush()
#
# t0 = time.time()
# generator = get_stream()
#
# while True:
#   try:
#     t0 = time.time()
#     time.sleep(0.101)
#     data = generator.next()
#     # print 'data:', data
#
#     # data2 = stream_exp2.read()
#     # print 'data2:', data2[:5]
#     # print 'len(data2)', len(data2)
#   except:
#     dq.stop()
#     break

import crappy
import time
import numpy as np

opendaq = crappy.technical.OpenDAQ(mode='streamer', channels=2, negative_channel=5, nsamples=254, sample_rate=200)
streamer = crappy.blocks.Streamer(sensor=opendaq, labels=['time(sec)', '2'])
grapher = crappy.blocks.Grapher(('time(sec)', '2'), length=50)
saver = crappy.blocks.Saver('/home/francois/Code/A_Projects/016_tests/fuka/oiureogy.csv', stamp='ouimonsieur')
labjack = crappy.technical.LabJack(actuator={'channel': 'TDAC0'})

crappy.link(in_block=streamer, out_block=grapher)
crappy.link(in_block=streamer, out_block=saver)
crappy.start()

volt = 0
x = np.linspace(0, 2 * np.pi, 100)
sine = np.sin(x)
period = 0
damping = 0.
t0 = 0

while True:
  try:
    for i in xrange(len(sine)):
      volt = sine[i] * damping * 10
      labjack.set_cmd(volt)
      time.sleep(0.02)
      damping = damping + 0.001 if damping < 1 else 1
      period += 1
  except:
    print 'nombre de periodes:', period
    opendaq.close()
    raise
