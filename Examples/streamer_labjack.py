# This program shows how to use labjack in streaming mode. Uncomment desired blocks.

import crappy

# Defining channels and sensor_dico
channels = ['AIN0', 'AIN1']
sensor_dico = {'mode': 'streamer',
               'scan_rate_per_channel': 10000,
               'scans_per_read': 1000,
               'channels': channels,
               'gain': [1, 2000]
               }

labjack = crappy.technical.LabJack(sensor=sensor_dico, verbose=True)
streamer = crappy.blocks.Streamer(sensor=labjack, labels=['time(sec)', 'Position(mm)', 'Force(N)'], mean=100)

# grapher = crappy.blocks.Grapher([('time(sec)', 'Force(N)')], length=10)
# grapher2 = crappy.blocks.Grapher([('time(sec)', 'Position(mm)')], length=10)
# saver = crappy.blocks.Saver('/home/francois/Essais/traction_ahmed/essai01.csv', stamp='date')

dash = crappy.blocks.Dashboard()
# crappy.link(streamer, grapher)
crappy.link(streamer, dash)
# crappy.link(streamer, grapher2)

crappy.start()
