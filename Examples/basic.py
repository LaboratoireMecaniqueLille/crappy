# coding: utf-8

import crappy

if __name__ == "__main__":
  print "Creating fake camera..."
  c = crappy.blocks.FakeCamera(512, 512)
  print "Creating sink..."
  s = crappy.blocks.Sink()

  print "Creating link and linking..."
  l = crappy.link(c, s, name='The_LINK')

  print "Starting..."
  crappy.start()
  print "Started !"
