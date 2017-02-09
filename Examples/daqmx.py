#coding: utf-8

import crappy

if __name__ == '__main__':
  s = crappy.blocks.DataReader(device='Dev2')
  g = crappy.blocks.Grapher(('t','V'),length=0, compacter=100)
  crappy.link(s,g)
  crappy.start()