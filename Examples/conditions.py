#coding: utf-8
"""
Demonstrates how to use conditions
"""

class My_condition():
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

import crappy

print("Available conditions:",crappy.condition.condition_list.keys())

if __name__ == "__main__":
  generator = crappy.blocks.Generator(path=[
    {'type':'constant','value':0, 'condition':'delay=2'},
    {'type':'constant','value':1, 'condition':'delay=2'}
    ]*20,spam=True)
  graph = crappy.blocks.Grapher(('t(s)','cmd'))
  smooth_graph = crappy.blocks.Grapher(('t(s)','cmd'))

  crappy.link(generator,graph)
  crappy.link(generator,smooth_graph,
      condition=[crappy.condition.Moving_avg(500),My_condition(5)])

  r = crappy.blocks.Reader('Trigged')

  crappy.link(generator,r,condition=crappy.condition.Trig_on_change('cycle'))

  crappy.start()
