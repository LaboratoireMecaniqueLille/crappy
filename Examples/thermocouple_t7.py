from __future__ import absolute_import
import crappy

if __name__ == "__main__":

  m = crappy.blocks.IOBlock("Labjack_t7",
      channels=dict(name='AIN0',thermocouple='K'),
      verbose=True,labels=['t(s)','T0'])

  g = crappy.blocks.Grapher(('t(s)','T0'))

  crappy.link(m,g)

  crappy.start()
