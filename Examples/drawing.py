# coding: utf-8

"""
Demonstrates the use of the Drawing block.

No hardware required.
"""

from time import sleep

import crappy


class TestBlock(crappy.Block):
  """A simple block to demonstrate the Drawing block."""

  def __init__(self, labels):
    super().__init__()
    self.labels = labels
    self.loops = 0

  def loop(self):
    r = {}
    for i, l in enumerate(self.labels):
      r[l] = self.loops + 15 * i
    self.send(r)
    self.loops += 1
    sleep(.2)


img = "data/Pad.png"

coord = [  # Coordinates for the thermocouples
         (185, 430),  # T1
         (145, 320),  # T2
         (105, 220),  # T3
         (720, 370),  # T4
         (720, 250),  # T5
         (720, 125),  # T6
         (1220, 410),  # T7
         (1260, 320),  # T8
         (1300, 230),  # T9
         ]

options = [{'type': 'dot_text',
            'coord': coord[i],
            'text': 'T{} = %.1f'.format(i + 1),
            'label': 'T' + str(i + 1)} for i in range(9)]

if __name__ == "__main__":
  s = TestBlock([d['label'] for d in options])

  options.append({"type": 'time', 'coord': (80, 1000)})

  d = crappy.blocks.Drawing(img, options, crange=[20, 300],
                            title="Temperatures")

  crappy.link(s, d)
  crappy.start()
