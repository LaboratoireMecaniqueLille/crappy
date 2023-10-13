# coding: utf-8

import crappy
import numpy.random as rd
from time import time


class CustomCam(crappy.camera.Camera):

  def __init__(self):

    super().__init__()
    self.low = 0
    self.high = 256
    self.color = False
    self.size = '480p'

    # Not elegant, but makes it possible to have 2 variants from a single file
    del self.low, self.high, self.color, self.size

    self._high = 255

  def open(self, low=0, high=256, color=False, size='480p'):

    self.add_scale_setting(name='low',
                           lowest=low,
                           highest=127,
                           getter=None,
                           setter=None,
                           default=0)

    self.add_scale_setting(name='high',
                           lowest=128,
                           highest=high,
                           getter=self._get_high,
                           setter=self._set_high,
                           default=256)

    self.add_bool_setting(name='color',
                          getter=None,
                          setter=None,
                          default=False)

    self.add_choice_setting(name='size',
                            choices=('240p', '480p', '720p'),
                            getter=None,
                            setter=None,
                            default='480p')

    self.set_all(low=low, high=high, color=color, size=size)

  def get_image(self):

    if self.size == '240p':
      if self.color:
        size = (240, 426, 3)
      else:
        size = (240, 426)
    elif self.size == '480p':
      if self.color:
        size = (480, 640, 3)
      else:
        size = (480, 640)
    else:
      if self.color:
        size = (720, 1280, 3)
      else:
        size = (720, 1280)

    img = rd.randint(low=self.low,
                     high=self.high,
                     size=size,
                     dtype='uint8')

    return time(), img

  def _get_high(self):

    return self._high

  def _set_high(self, value):

    self._high = value


if __name__ == '__main__':

  cam = crappy.blocks.Camera('CustomCam',
                             config=True,
                             display_images=True,
                             displayer_framerate=30,
                             freq=30,
                             save_images=False)

  stop = crappy.blocks.StopButton()

  crappy.start()
