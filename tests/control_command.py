import crappy

sensor = {
  'channels': 'AIN0',
  'gain': 1,
  'offset': 0,
  'resolution': 1,
  'chan_range': 10,
  'mode': 'single'
}
actuator = {
  'channel': 'TDAC0',
  'gain': 10,
  'offset': 0
}
# wave = crappy.blocks.SignalGenerator(path=[{'step': 0, 'waveform': 'sinus', 'time': 1, 'phase': 0, 'amplitude': 10,
#                                               'freq': 1, 'offset': 0},
#                                              {'step': 0, 'waveform': 'sinus', 'time': 1, 'phase': 0, 'amplitude': 5,
#                                               'freq': 10, 'offset': 0},
#                                              {'step': 0, 'waveform': 'triangle', 'time': 1, 'phase': 0, 'amplitude': 2,
#                                               'freq': 100, 'offset': 0},
#                                              {'step': 0, 'waveform': 'square', 'time': 1, 'phase': 0, 'amplitude': 10,
#                                               'freq': 5, 'offset': 0},
#                                              {'step': 0, 'waveform': 'sinus', 'time': 1, 'phase': 0, 'amplitude': 10,
#                                               'freq': 120, 'offset': 0}
#                                              ], send_freq=800, repeat=True)
opendaq = crappy.technical.OpenDAQ()
labjack = crappy.technical.LabJack(sensor=sensor, actuator=actuator, device='t7')


wave_opendaq = crappy.blocks.WaveGenerator(wave_frequency=1)
wave_labjack = crappy.blocks.WaveGenerator(wave_frequency=1)
pdaq = True

cc_opendaq = crappy.blocks.ControlCommand(opendaq, compacter=100, verbose=not pdaq)
cc_labjack = crappy.blocks.ControlCommand(labjack, compacter=100, verbose=pdaq)

grapher_opendaq = crappy.blocks.Grapher(('time(sec)', 1), ('time(sec)', 'signal'), length=20)
grapher_labjack = crappy.blocks.Grapher(('time(sec)', 'AIN0'), ('time(sec)', 'signal'), length=20)
# sink1 = crappy.blocks.Sink()
# sink1 = crappy.blocks.Dashboard()
# sink2 = crappy.blocks.Dashboard()
# sink2 = crappy.blocks.Sink()
crappy.link(wave_labjack, cc_labjack)
crappy.link(wave_opendaq, cc_opendaq)
# crappy.link(cc_labjack, sink1)
# crappy.link(cc_opendaq, sink2)

#
crappy.link(cc_labjack, grapher_labjack)
crappy.link(cc_opendaq, grapher_opendaq)

crappy.start()
