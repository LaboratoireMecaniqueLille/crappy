# Un beau programme pour tester la commande en meme temps que l'acquistion
import crappy
import time

sensor = {
    'channels': ['AIN0', 'AIN1', 'AIN2', 'AIN3'],
    'gain': 1,
    'offset': 0,
    'resolution': 0,
    'chan_range': 10,
    'mode': 'thermocouple'
}
actuator = {
    'channel': 'DAC0',
    'gain': 1,
    'offset': 0
}

crappy.blocks.MasterBlock.instances = []
labjack = crappy.technical.LabJack(sensor=sensor, actuator=actuator)
measurebystep = crappy.blocks.MeasureByStep(sensor=labjack)
compacter = crappy.blocks.Compacter(2)
saver = crappy.blocks.Saver('/home/francois/freq.csv', stamp='yes')

link1 = crappy.links.Link()
link2 = crappy.links.Link()

measurebystep.add_output(link1)
compacter.add_input(link1)

compacter.add_output(link2)
saver.add_input(link2)

t0 = time.time()
try:
    for instance in crappy.blocks.MasterBlock.instances:
        instance.t0 = t0
    for instance in crappy.blocks.MasterBlock.instances:
        instance.start()
except:
    for instance in crappy.blocks.MasterBlock.instances:
        instance.stop()

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
