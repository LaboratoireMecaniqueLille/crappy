import crappy

gen = crappy.blocks.Generator([{'type':'ramp','condition':None,
  'speed':1}],freq=500)
graph = crappy.blocks.Grapher(('t(s)','i_cmd'),maxpt=10,interp=True)
crappy.link(gen,graph,modifier=crappy.modifier.Integrate('cmd'))

crappy.start()
