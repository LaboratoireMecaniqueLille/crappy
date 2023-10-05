# coding: utf-8

import crappy

if __name__ == '__main__':

  cam = crappy.blocks.Camera('FakeCamera',
                             config=True,
                             display_images=True,
                             displayer_framerate=30,
                             save_images=False,
                             freq=40)

  stop = crappy.blocks.StopButton()

  crappy.start()
