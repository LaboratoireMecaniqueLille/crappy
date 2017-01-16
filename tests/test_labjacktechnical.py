# Un beau programme pour tester la commande en meme temps que l'acquistion
import crappy
import time

sensor = {
  'channels': range(1),
  'gain': 1,
  'offset': 0,
  'resolution': 9,
  'chan_range': 10,
  'mode': 'thermocouple'
}
actuator = {
  'channel': 'DAC0',
  'gain': 1,
  'offset': 0
}

labjack = crappy.technical.LabJack(sensor=sensor, actuator=actuator, device='t7')
measurebystep = crappy.blocks.MeasureByStep(sensor=labjack, verbose=True, compacter=100, freq=None)
# saver = crappy.blocks.Saver('/home/francois/freq.csv', stamp='yes')
dash = crappy.blocks.Dashboard()

crappy.link(measurebystep, dash)
crappy.start()


# print 'bien ouvert'
# volt = 0
# compteur = 0
# t0 = time.time()
# while True:
#     compteur += 1
#     results = labjack.get_data()
#     if compteur % 2 == 0:
#         volt = 1
#     else:
#         volt = 0
#     labjack.set_cmd(volt)
#
#     if time.time() - t0 > 10:
#         labjack.close()
#         tfinal = time.time()
#         break
#
# elapsed = tfinal - t0
# print 'Freq: %.2f' % (compteur / elapsed)
