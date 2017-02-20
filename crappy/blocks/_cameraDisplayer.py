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
    if cv:
      self.loop = self.loop_cv
      self.begin = self.begin_cv
      self.finish = self.finish_cv
    else:
      self.loop = self.loop_mpl
      self.begin = self.begin_mpl
      self.finish = self.finish_mpl

  def begin_mpl(self):
    import matplotlib.pyplot as plt
    self.plt = plt
    self.plt.ion()
    fig = self.plt.figure()
    fig.add_subplot(111)

  def loop_mpl(self):
    frame = self.inputs[0].recv()['frame']
    self.plt.imshow(frame, cmap='gray')
    self.plt.pause(0.001)
    self.plt.show()

  def begin_cv(self):
    import cv2
    self.cv2 = cv2
    try:
      flags = self.cv2.WINDOW_NORMAL | self.cv2.WINDOW_KEEPRATIO
    except AttributeError:  # WINDOW_KEEPRATIO is not implemented in all opencv versions...
      flags = self.cv2.WINDOW_NORMAL
    self.cv2.namedWindow(self.title, flags)

  def loop_cv(self):
    data = self.inputs[0].recv_delay(self.delay)['frame'][-1]
    self.cv2.imshow(self.title,data)
    self.cv2.waitKey(1)

  def finish_mpl(self):
    self.plt.close('all')

  def finish_cv(self):
    self.cv2.destroyAllWindows()
