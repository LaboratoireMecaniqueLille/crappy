#coding: utf-8
"""
Demonstrates how to use modifiers

Number of modifiers are already defined une crappy.modifiers, but it can
also be a function or a class with the .evaluate() method
"""
import crappy


class My_modifier():
  def __init__(self,offset):
    self.offset = offset

  def evaluate(self,data):
    """
    Remember: data is ALWAYS a dict
    returning None will drop the data
    """
    for k in data:
      if k != 't(s)': # Move everything except the time
        data[k] += self.offset
    return data # Do not forget to return it!


if __name__ == "__main__":
  generator = crappy.blocks.Generator(path=[
    {'type':'constant','value':0, 'condition':'delay=2'},
    {'type':'constant','value':1, 'condition':'delay=2'}
      ]*20,spam=True)
  graph = crappy.blocks.Grapher(('t(s)','cmd'))
  smooth_graph = crappy.blocks.Grapher(('t(s)','cmd'))

  crappy.link(generator,graph)
  # We add a moving average to smooth the data
  # and our custom condition that adds and offset of 5
  crappy.link(generator,smooth_graph,
      modifier=[crappy.modifier.Moving_avg(500),My_modifier(5)])

  # This block will simply print "Trigged" followed by the received data
  r = crappy.blocks.Reader('Trigged')

  # Only forward data when the label "cycle" changed its value
  crappy.link(generator,r,modifier=crappy.modifier.Trig_on_change('cycle'))

  crappy.start()
