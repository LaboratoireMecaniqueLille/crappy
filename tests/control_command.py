import crappy

sensor = {
  'channels': range(1),
  'gain': 1,
  'offset': 0,
  'resolution': 1,
  'chan_range': 10,
  'mode': 'single'
}
actuator = {
  'channel': 'TDAC0',
  'gain': 1,
  'offset': 0
}
opendaq = crappy.technical.OpenDAQ()
signal = crappy.blocks.SignalGenerator(path=[{'step': 0, 'waveform': 'sinus', 'time': 1, 'phase': 0, 'amplitude': 10,
                                              'freq': 1, 'offset': 0},
                                             {'step': 0, 'waveform': 'sinus', 'time': 1, 'phase': 0, 'amplitude': 5,
                                              'freq': 10, 'offset': 0},
                                             {'step': 0, 'waveform': 'triangle', 'time': 1, 'phase': 0, 'amplitude': 2,
                                              'freq': 100, 'offset': 0},
                                             {'step': 0, 'waveform': 'square', 'time': 1, 'phase': 0, 'amplitude': 10,
                                              'freq': 5, 'offset': 0},
                                             {'step': 0, 'waveform': 'sinus', 'time': 1, 'phase': 0, 'amplitude': 10,
                                              'freq': 120, 'offset': 0}
                                             ], send_freq=800, repeat=True)
labjack = crappy.technical.LabJack(sensor=sensor, actuator=actuator, device='t7')

measurebystep = crappy.blocks.MeasureByStep(sensor=opendaq, verbose=True, compacter=100, freq=1000)
# grapher = crappy.blocks.Grapher(('time(sec)', 'AIN0'), ('time(sec)', 'signal'), length=10)
grapher2 = crappy.blocks.Grapher(('time(sec)', 1), length=10)
cc = crappy.blocks.ControlCommand(labjack, compacter=100, verbose=True)
dash = crappy.blocks.Dashboard(nb_digits=1)

crappy.link(signal, cc)
# crappy.link(cc, grapher)
crappy.link(measurebystep, grapher2)
crappy.link(measurebystep, dash)
crappy.start()
