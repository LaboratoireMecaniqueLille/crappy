# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup CameraDisplayer CameraDisplayer
# @{

## @file _cameraDisplayer.py
# @brief Simple images displayer. Can be paired with StreamerCamera
#   Use cv=False to use the old, inefficient and deprecated version
#   NOTE: You need to use one displayer block per window (in other words, you can only attach one input to the diplayer)
#
# @author Corentin Martel, Robin Siemiatkowski, Victor Couty
# @version 0.1
# @date 05/07/2016

from _masterblock import MasterBlock
from time import time


class CameraDisplayer(MasterBlock):
  """
  Simple images displayer. Can be paired with StreamerCamera
  Use cv=False to use the old, inefficient and deprecated version
  NOTE: You need to use one displayer block per window (in other words, you can only attach one input to the diplayer)
  """

  def __init__(self, framerate=5, cv=True, title='Displayer'):
    super(CameraDisplayer, self).__init__()
    if framerate is None:
      self.delay = 0
    else:
      self.delay = 1. / framerate  # Framerate (fps)
    self.cv = cv
    self.title = title

  def main(self):
    try:
      if not self.cv:
        import matplotlib.pyplot as plt
        plt.ion()
        fig = plt.figure()
        fig.add_subplot(111)
        while True:
          # print "top loop"
          frame = self.inputs[0].recv()['frame']
          # print frame.shape
          # if frame None:
          # print frame[0][0]
          plt.imshow(frame, cmap='gray')
          plt.pause(0.001)
          plt.show()
      else:
        import cv2
        data = 0
        try:
          flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
        except AttributeError:  # WINDOW_KEEPRATIO is not implemented in all opencv versions...
          flags = cv2.WINDOW_NORMAL
        cv2.namedWindow(self.title, flags)
        while True:
          t1 = time()
          while data is not None:  # To flush the pipe...
            last = data  # Save the latest non-None value
            data = self.inputs[0].recv(False)  # ... use non-blocking recv until pipe is empty
          if last is not 0:  # If we received something (ie the last non-None value is not 0)
            cv2.imshow('Displayer', last['frame'])
            cv2.waitKey(1)
          data = 0  # A default value to check if we received something
          while time() - t1 < self.delay:
            data = self.inputs[0].recv()

    except KeyboardInterrupt:
      if self.cv:
        cv2.destroyAllWindows()
      else:
        plt.close('all')
    except Exception as e:
      if self.cv:
        cv2.destroyAllWindows()
      else:
        plt.close('all')
      print "Exception in CameraDisplayer : ", e
      raise
