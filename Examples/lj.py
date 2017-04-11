import crappy

m = crappy.blocks.MeasureByStep('Labjack_T7',labels=['t(s)','AIN0'],verbose=True)

g = crappy.blocks.Grapher(('t(s)','AIN0'))

crappy.link(m,g)

crappy.start()
