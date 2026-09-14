# coding: utf-8

import crappy

if __name__ == '__main__':

  camera = crappy.blocks.vision.CameraSource('FakeCamera',
                                             config=True,
                                             freq=40)

  displayer = crappy.blocks.vision.ImageDisplayer(framerate=30,
                                                  freq=40)

  stop = crappy.blocks.StopButton()

  crappy.img_link(camera, displayer)

  crappy.start()
