# coding: utf-8

import crappy
from cv2 import imshow, waitKey

if __name__ == '__main__':

  cam = crappy.camera.FakeCamera()
  cam.open(width=1280, height=720, speed=100, fps=50)
  img = cam.get_image()[1]

  imshow('picture', img)
  waitKey(3000)
