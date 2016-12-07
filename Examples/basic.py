#coding: utf-8

import crappy

if __name__ == "__main__":

	print "Creating fake camera..."
	c = crappy.blocks.FakeCamera(512,512)
	print "Creating sink..."
	s = crappy.blocks.Sink()

	print "Creating link..."
	l = crappy.links.Link()

	print "Linking..."
	c.add_output(l)
	s.add_input(l)

	print "Starting..."
	crappy.start()
	print "Started !"
