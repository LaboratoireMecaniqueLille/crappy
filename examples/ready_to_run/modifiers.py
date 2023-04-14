# coding: utf-8

"""
Demonstrates how to use modifiers.

Number of modifiers are already defined in ``crappy.modifiers``, but it can
also be a function or any class containing the ``.evaluate()`` method.

No hardware required.
"""

import crappy


# Example of class used as a Modifier
class My_offset_modifier(crappy.Modifier):
  def __init__(self, offset):
    super().__init__()
    self.offset = offset

  def __call__(self, data):
    """Method returning the modified values.

    Remember: data is ALWAYS a :obj:`dict`.
    Returning :obj:`None` will drop the data.
    """

    for k in data:
      if k != 't(s)':  # Move everything except the time
        data[k] += self.offset
    return data  # Do not forget to return it!


# Example of function used as a modifier
def mul_by_10(data):
  data['cmd'] *= 10
  return data


if __name__ == "__main__":
  generator = crappy.blocks.Generator(path=[
    {'type': 'Constant', 'value': 0, 'condition': 'delay=2'},
    {'type': 'Constant', 'value': 1, 'condition': 'delay=2'}
      ] * 20, spam=True)
  graph = crappy.blocks.Grapher(('t(s)', 'cmd'))
  smooth_graph = crappy.blocks.Grapher(('t(s)', 'cmd'))

  crappy.link(generator, graph)
  # We add a moving average to smooth the data
  # and our custom condition that adds and offset of 5
  crappy.link(generator, smooth_graph,
              # The modifiers will be applied in the order of the list
              modifier=[
                # Integrated modifier, will average the values on 100 points
                crappy.modifier.MovingAvg(100),
                # Will add an offset
                My_offset_modifier(5),
                # Will multiply the result by 10
                mul_by_10])

  # This block will simply print "Triggered" followed by the received data
  r = crappy.blocks.LinkReader('Triggered')

  # Only forward data when the label "cycle" changed its value
  crappy.link(generator, r,
              modifier=crappy.modifier.TrigOnChange('cycle'))

  crappy.start()
