
import crappy


m = crappy.blocks.IOBlock('nidaqmx',
    labels=['t(s)','chan0'],verbose=True)

g = crappy.blocks.Grapher(('t(s)','chan0'))

crappy.link(m,g)

crappy.start()
