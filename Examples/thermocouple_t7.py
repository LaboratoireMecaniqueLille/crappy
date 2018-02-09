from __future__ import absolute_import
import crappy

chan = [0,2,3,4,5]

if __name__ == "__main__":

  m = crappy.blocks.IOBlock("Labjack_t7",
      channels=[dict(name='AIN%d'%i,thermocouple='K') for i in chan],
      verbose=True,labels=['t(s)']+['T%d'%i for i in chan])

  g = crappy.blocks.Grapher(*[('t(s)','T%d'%i) for i in chan])

  #crappy.link(m,g)
  crappy.link(m,g,condition=crappy.condition.Moving_avg(10))

  crappy.start()
