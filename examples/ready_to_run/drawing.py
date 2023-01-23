# coding: utf-8

"""
Demonstrates the use of the Drawing block.

No hardware required.
Requires the cv2 module to be installed.
"""

from typing import List
import crappy


class TestBlock(crappy.Block):
  """A simple block for demonstrating the use of the Drawing block.

  Simply updates the values on the Drawing at each loop.
  """

  def __init__(self, labels: List[str]) -> None:
    """Sets the args and initializes the parent class."""

    super().__init__()
    self.freq = 5

    self._labels = labels
    self._loops = 0

  def loop(self) -> None:
    """At each loop, returns the updated dict of values to display."""

    ret = {}
    for i, label in enumerate(self._labels):
      ret[label] = self._loops + 15 * i
    self.send(ret)
    self._loops += 1


if __name__ == "__main__":

  # The path to the background image
  img_path = crappy.resources.paths['pad']

  # The coordinates of the thermocouples on the drawing
  coord = [(185, 430), (145, 320), (105, 220), (720, 370), (720, 250),
           (720, 125), (1220, 410), (1260, 320), (1300, 230)]

  # The dict containing the information on the elements to display on the
  # Drawing. Here we add 'dot_text' elements.
  options = [{'type': 'dot_text',
              'coord': coord[i],
              'text': f'T{i + 1} = %.1f',
              'label': f'T{i + 1}'} for i in range(len(coord))]

  # The block that will update the values of the 'dot_text' elements on the
  # Drawing. We simply give it a list containing the label of each element.
  update_block = TestBlock([d['label'] for d in options])

  # Now adding a 'time' element to the Drawing. The 'text' and 'label' keys
  # are not mandatory and do not affect the Drawing.
  options.append({'type': 'time',
                  'coord': (80, 1000),
                  'text': '',
                  'label': ''})

  # The block containing the Drawing
  drawing = crappy.blocks.Drawing(image_path=img_path, draw=options,
                                  color_range=(20, 300), title="Temperatures")

  # Linking and starting the blocks
  crappy.link(update_block, drawing)
  crappy.start()
