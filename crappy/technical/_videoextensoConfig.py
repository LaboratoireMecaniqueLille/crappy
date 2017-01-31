# coding: utf-8
##  @addtogroup technical
# @{

##  @defgroup camerainit CameraInit
# @{

## @file _cameraInit.py
# @brief Gui to open and initialize a camera device.
#
# @author Robin Siemiatkowski
# @version 0.1
# @date 29/06/2016

import numpy as np
# import time
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import matplotlib.patches as mpatches
import cv2

rectprops = dict(facecolor='red', edgecolor='red', alpha=0.5, fill=True)
from skimage.segmentation import clear_border
from skimage.morphology import label, erosion, square, dilation
from skimage.measure import regionprops

try:
  from skimage.filters import threshold_otsu, rank  # load newer version
except:
  from skimage.filter import threshold_otsu, rank  # load deprecated version


class _CameraInit:
  def __init__(self, camera, videoextenso):
    # videoextenso = {'enabled':True, 'white_spot':True, 'border':4,'xoffset':0,'yoffset':0,
    # 'width':2048,'height':2048}
    self.videoextenso = videoextenso
    self.cam = camera
    self.rect = {}
    rat = 0.7
    Width = 7
    Height = 7.
    self._fig = plt.figure(figsize=(Height, Width))
    self._axim = self._fig.add_axes([0.15, 0.135, rat, rat * (Height / Width)])  # Image frame
    self._axim.set_autoscale_on(True)
    self._cax = self._fig.add_axes([0.17 + rat, 0.135, 0.02, rat * (Height / Width)])  # colorbar frame
    self._axhist = self._fig.add_axes([0.15, (0.17 + rat), rat, 0.1])  # histogram frame
    axcolor = 'lightgoldenrodyellow'
    #self.cam.open(width=self.videoextenso['width'], height=self.videoextenso['height'],
    #             yoffset=self.videoextenso['yoffset'], xoffset=self.videoextenso['xoffset'])
    """
    self.cam.height = self.videoextenso['height']
    self.cam.width = self.videoextenso['width']
    self.cam.yoffset = self.videoextenso['yoffset']
    self.cam.xoffset = self.videoextenso['xoffset']
"""

    #if self.cam.gain is not None:
    if 'gain' in self.cam.available_settings:
      self._axGain = plt.axes([0.15, 0.07, rat, 0.03], axisbg=axcolor)
      self._sGain = Slider(self._axGain, 'Gain', -1, 6, valinit=self.cam.gain)
      self._sGain.on_changed(self.update_gain)

    # define buttons here
    Closeax = plt.axes([0.01, (0.30 + rat) / 2, 0.07, 0.05])  # define size and position
    self._CloseButton = Button(Closeax, 'Save', color='blue', hovercolor='0.975')
    # define cursors here


    # Exposition max = 1000000 # define slider with previous position and size
    if 'exposure' in self.cam.available_settings:
      axExp = plt.axes([0.15, 0.02, rat, 0.03], axisbg=axcolor)  # define position and size
      self._sExp = Slider(axExp, 'Exposure', 200, 50000, valinit=self.cam.exposure)
      self._sExp.on_changed(self.update_exposure)

    self._CloseButton.on_clicked(self.close)

    # initialising the histogram
    if camera.name.lower() == 'ximea':
      from ..sensor import ximeaModule as xi
      if self.cam.ximea.get(xi.CAP_PROP_XI_DATA_FORMAT) == 0 or self.cam.ximea.get(
          xi.CAP_PROP_XI_DATA_FORMAT) == 5:
        self.x = np.arange(0, 256, 4)
      elif self.cam.ximea.get(xi.CAP_PROP_XI_DATA_FORMAT) == 1 or self.cam.ximea.get(
          xi.CAP_PROP_XI_DATA_FORMAT) == 6:
        self.x = np.arange(0, 1024, 4)
    else:
      self.x = np.arange(0, 256, 4)

    hist = np.ones(np.shape(self.x))
    frame = self.cam.get_image()
    # print "type frame : " , type(frame)

    self._axhist.set_xlim([0, max(self.x)])  # set histogram limit in x...
    self._axhist.set_ylim([0, 1])  # ... and y
    # im = self.axim.imshow(frame,cmap=plt.cm.gray,interpolation='nearest') # display the first
    self._li, = self._axhist.plot(self.x, hist)  # plot first histogram
    self._im = self._axim.imshow(frame, cmap=plt.cm.gray, interpolation='nearest')  # display the first image
    cb = self._fig.colorbar(self._im, cax=self._cax)  # plot colorbar
    self._cax.axis('off')

  # self.width = self.cam.width
  # self.height = self.cam.height
  # self.yoffset = self.cam.yoffset
  # self.xoffset = self.cam.xoffset

  def start(self):
    if self.videoextenso:
      def toggle_selector(self, event):
        toggle_selector.RS.set_active(False)

      toggle_selector.RS = RectangleSelector(self._axim.properties().get('axes'), self.zoi_selection,
                                             drawtype='box', useblit=True,
                                             button=[1, 3],  # don't use middle button
                                             minspanx=5, minspany=5, rectprops=rectprops,
                                             spancoords='pixels')
    ani = animation.FuncAnimation(self._fig, self.get_frame, interval=50, frames=20, blit=False)
    try:
      plt.show()
    except AttributeError:
      pass

  def zoi_selection(self, eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    xmin = round(min(x1, x2))
    xmax = round(max(x1, x2))
    ymin = round(min(y1, y2))
    ymax = round(max(y1, y2))
    # update dimension of the image:
    self.height = (ymax - ymin)
    self.width = (xmax - xmin)
    self.yoffset = ymin
    self.xoffset = xmin
    # print "1"
    # camera INIT with ZOI selection
    # the following is to initialise the spot detection
    image = self.cam.get_image()
    # plt.imsave("/home/corentin/Bureau/image_originale.tiff",image)
    croped_image = image[self.yoffset:self.height + self.yoffset, self.xoffset:self.xoffset + self.width]
    image = croped_image
    # plt.imsave("/home/corentin/Bureau/image.tiff",image)
    image = rank.median(image, square(
      15))  # median filter to smooth the image and avoid little reflection that may appear as spots.
    self.thresh = threshold_otsu(image)  # calculate most effective threshold
    bw = image > self.thresh
    # print "1"
    # applying threshold
    if not (self.videoextenso['white_spot']):
      bw = (1 - bw).astype(np.uint8)
    # still smoothing
    bw = dilation(bw, square(3))
    bw = erosion(bw, square(3))
    # plt.imsave("/home/corentin/Bureau/bw.tiff",bw)
    # Remove artifacts connected to image border
    cleared = bw.copy()
    clear_border(cleared)
    # Label image regions
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    # plt.imsave("/home/corentin/Bureau/label_image.tiff",label_image)
    # Create the empty vectors for corners of each ZOI
    regions = regionprops(label_image)
    print [region.area for region in regions]
    # mean_area=np.mean[region.area for region in regions]
    regions = [region for region in regions if region.area > 200]
    self.NumOfReg = len(regions)
    if self.NumOfReg:
      print " Spots detected in camerainit: ", self.NumOfReg
      # smoothing=1
      self.minx = np.empty([self.NumOfReg, 1])
      self.miny = np.empty([self.NumOfReg, 1])
      self.maxx = np.empty([self.NumOfReg, 1])
      self.maxy = np.empty([self.NumOfReg, 1])
      self.Points_coordinates = np.empty([self.NumOfReg, 2])
      # Definition of the ZOI and initialisation of the regions border
      i = 0
      for i, region in enumerate(regions):  # skip small regions
        # if region.area > 100:
        self.minx[i], self.miny[i], self.maxx[i], self.maxy[i] = region.bbox

        # for k in range(smoothing):
        # print k
      image = self.cam.get_image()
      # plt.imsave("/home/corentin/Bureau/image_originale.tiff",image)
      croped_image = image[self.yoffset:self.height + self.yoffset, self.xoffset:self.xoffset + self.width]
      image = croped_image
      self.thresh = threshold_otsu(
        image)  # you have to re-evaluate the threashold here to have the same as you will after
      if self.NumOfReg == 1:
        self.Points_coordinates[i, 0], self.Points_coordinates[i, 1], self.minx[i, 0], self.miny[i, 0], self.maxx[
          i, 0], self.maxy[i, 0], self.L0x, self.L0y = self.barycenter_opencv(
          image[self.minx[i, 0] - 1:self.maxx[i, 0] + 1, self.miny[i, 0] - 1:self.maxy[i, 0] + 1],
          self.minx[i, 0] - 1, self.miny[i, 0] - 1)
      else:
        for i in range(0, self.NumOfReg):  # find the center of every region
          self.Points_coordinates[i, 0], self.Points_coordinates[i, 1], self.minx[i, 0], self.miny[i, 0], \
          self.maxx[i, 0], self.maxy[i, 0] = self.barycenter_opencv(
            image[self.minx[i, 0] - 1:self.maxx[i, 0] + 1, self.miny[i, 0] - 1:self.maxy[i, 0] + 1],
            self.minx[i, 0] - 1, self.miny[i, 0] - 1)

      self.minx += self.yoffset
      self.maxx += self.yoffset
      self.miny += self.xoffset
      self.maxy += self.xoffset

      image = self.cam.get_image()  # read a frame
      print "type of frame: ", type(image)

      minx_ = self.minx.min()
      miny_ = self.miny.min()
      maxx_ = self.maxx.max()
      maxy_ = self.maxy.max()

      minx = self.minx - minx_
      maxx = self.maxx - minx_
      miny = self.miny - miny_
      maxy = self.maxy - miny_

      # print "3"
      frame = image[minx_:maxx_, miny_:maxy_]
      self.rec = {}
      center = {}
      if self.videoextenso['white_spot']:
        color = 255
      else:
        color = 0

      if len(self.rect) > 0:
        for i in range(len(self.rect)):
          # print 'test'
          self.rect[i].remove()
        self.rect = {}
      # evaluating the reduction ratio to enlarge the rectangles
      ratx = 1#self.videoextenso['height'] * 1. / image.shape[0] #self.cam.height
      raty = 1#self.videoextenso['width'] * 1. / image.shape[1] #self.cam.width
      for i in range(self.NumOfReg):
      # For each region, plots the rectangle around the spot and a cross at the center
        print self.maxx[i], self.maxy[i]
        self.rect[i] = mpatches.Rectangle((self.miny[i] * raty, self.minx[i] * ratx), (maxy[i] - miny[i]) * raty,
                                          (maxx[i] - minx[i]) * ratx, fill=False, edgecolor='red', linewidth=1)
        self.rec[i] = self._axim.get_axes().add_patch(self.rect[i])
    else:
      print "No spot detected."

  def barycenter_opencv(self, image, minx, miny):
    """
    computatition of the barycenter (moment 1 of image) on ZOI using OpenCV
    White_Mark must be True if spots are white on a dark material
    """
    # The median filter helps a lot for real life images ...
    # print "5"
    # print image.shape
    # self.thresh=threshold_otsu(image)
    # plt.imsave("test1.tiff", image)
    bw = cv2.medianBlur(image, 5) > self.thresh
    if not (self.videoextenso['white_spot']):
      bw = 1 - bw
    M = cv2.moments(bw * 255.)
    try:
      Px = M['m01'] / M['m00']
      Py = M['m10'] / M['m00']
    except Exception as e:
      print "ERROR: ", e
      import time
      time.sleep(0.1)
      return self.barycenter_opencv(image, minx, miny)
    if self.NumOfReg == 1:
      a = M['mu20'] / M['m00']
      b = -M['mu11'] / M['m00']
      c = M['mu02'] / M['m00']
      l1 = 0.5 * ((a + c) + np.sqrt(4 * b ** 2 + (a - c) ** 2))
      l2 = 0.5 * ((a + c) - np.sqrt(4 * b ** 2 + (a - c) ** 2))
      minor_axis = 4 * np.sqrt(l2)
      major_axis = 4 * np.sqrt(l1)
      if (a - c) == 0:
        if b > 0:
          theta = -np.pi / 4
        else:
          theta = np.pi / 4
      else:
        theta = 0.5 * np.arctan2(2 * b, (a - c))
      # print "min,maj,theta :" ,minor_axis,major_axis,theta
      Lx = max(np.abs(major_axis * np.cos(theta)), np.abs(minor_axis * np.sin(theta)))
      Ly = max(np.abs(major_axis * np.sin(theta)), np.abs(minor_axis * np.cos(theta)))
    # print Dx, Dy, Px,Py
    # Px=Dx
    # Py=Dy
    # print "Dx0,Dy0 : ", Dx,Dy
    # we add minx and miny to go back to global coordinate:
    # else:
    Px += minx
    Py += miny
    miny_, minx_, h, w = cv2.boundingRect(
      (bw * 255).astype(np.uint8))  # cv2 returns x,y,w,h but x and y are inverted
    maxy_ = miny_ + h
    maxx_ = minx_ + w
    # Determination of the new bounding box using global coordinates and the margin
    border = self.videoextenso['border']
    minx = minx - border + minx_
    miny = miny - border + miny_
    maxx = minx + border + maxx_
    maxy = miny + border + maxy_
    if self.NumOfReg == 1:
      return Px, Py, minx, miny, maxx, maxy, Lx, Ly
    else:
      return Px, Py, minx, miny, maxx, maxy

  def update_exposure(self, val):  # this function updates the exposure
    self.cam.exposure = self._sExp.val
    self._fig.canvas.draw_idle()

  def update_gain(self, val):  # this function updates the exposure and gain values
    self.cam.gain = self._sGain.val
    self._fig.canvas.draw_idle()

  def close(self, event):
    try:
      if self.NumOfReg == 4 or self.NumOfReg == 2:
        self.L0x = self.Points_coordinates[:, 0].max() - self.Points_coordinates[:, 0].min()
        self.L0y = self.Points_coordinates[:, 1].max() - self.Points_coordinates[:, 1].min()
      print "L0 saved! : ", self.L0y, self.L0x
    except AttributeError:  # if no selected Points_coordinates
      print "no points selected"

  # Main
  def get_frame(self, i):
    frame = self.cam.get_image()  # read a frame
    # x=np.arange(0,2048,4)
    if i == 1:
      self._cax.axis('on')
      self._im.set_data(frame)  # change previous image by new frame
      self._im.set_clim([frame.min(), frame.max()])  # re-arrange colorbar limits
      histogram, bins = np.histogram(frame.ravel(), len(self.x), [0, max(self.x)])  # evalute new histogram
      histogram = np.sqrt(histogram)  # this operation aims to improve the histogram visibility (avoid flattening)
      self._li.set_ydata(
        (histogram - histogram.min()) / (histogram.max() - histogram.min()))  # change previous histogram
      # self._axim.set_autoscale_on(True)
      return self._cax, self._axim, self._axhist  # return the values that need to be updated

  def get_configuration(self):
    if self.videoextenso:
      # print "thresh in camera init :" ,self.thresh
      return self.minx, self.maxx, self.miny, self.maxy, self.NumOfReg, self.L0x, self.L0y, self.thresh, self.Points_coordinates
    else:
      return int(self.cam.exposure), self.cam.gain, int(self.cam.width), int(self.cam.height), int(
        self.cam.xoffset), int(self.cam.yoffset)


def get_camera_config(cam, videoExtenso={}, send_pipe=None):
  """
  Function to open and configure a camera device.
  Args:
      cam: instance of a camera class.
      videoExtenso: dictionnary to enable or disable the videoExtenso initialization.
      send_pipe:
  """
  d = _CameraInit(cam, videoExtenso)
  d.start()
  #d.cam.close()
  plt.close()
  if send_pipe:
    send_pipe.send(d.get_configuration())
  else:
    return d.get_configuration()

from ._cameraConfig import Camera_config

def ve_config(cam):
  return Ve_config().config(cam)

class Ve_config(Camera_config):
  def __init__(self):
    Camera_config.__init__(self)

  def config(self,camera):
    pass
