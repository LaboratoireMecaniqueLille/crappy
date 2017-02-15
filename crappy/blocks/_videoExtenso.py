# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup VideoExtenso VideoExtenso
# @{

## @file _videoExtenso.py
# @brief Detects spots (1,2 or 4) on images, and evaluate the deformations
# @authors Robin Siemiatkowski, Corentin Martel
# @version 0.1
# @date 13/07/2016

import numpy as np
import time
import cv2
import SimpleITK as sitk  # only for testing
import os
from multiprocessing import Process, Pipe

from ._masterblock import MasterBlock

from ..links._link import TimeoutError
from ..technical import TechnicalVideoExtenso as tve

try:
  from skimage.filters import threshold_otsu  # load newest version
except ImportError:
  from skimage.filter import threshold_otsu  # load deprecated version
from sys import stdout

np.set_printoptions(threshold='nan', linewidth=500)

def plotter(plot_pipe_recv):
  """
  Wait for data and plot a frame in an opencv window.

  Args:
      plot_pipe_recv: Pipe which receive data to plot (numpy.ndarray).

  Returns:
      void return function, first it opens a new opencv windows which plot the
      picture.

  """
  data = plot_pipe_recv.recv()  # receiving data
  NumOfReg = data[0]
  minx = data[1]
  maxx = data[2]
  miny = data[3]
  maxy = data[4]
  Points_coordinates = data[5]
  #L0x = data[6]
  #L0y = data[7]
  frame = data[8]
  white_spot = data[9]
  if white_spot:
    color = 255
  else:
    color = 0
  cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
  # For each region, plots the rectangle around the spot and a cross
  # at the center
  for i in range(0, NumOfReg):
    frame = cv2.rectangle(frame, (miny[i], minx[i]),
                         (maxy[i] - 1, maxx[i] - 1), (color, 0, 0), 1)
    frame = cv2.circle(frame, (int(Points_coordinates[i, 1]),
                         int(Points_coordinates[i, 0])), 1,
                       (255 - color, 0, 0), -1)
  cv2.imshow('frame', frame)
  cv2.waitKey(1)
  # for every round, receive data, correct the positions of the
  # rectangles/centers and the values of Lx/Ly ,
  # and refresh the plot.
  while True:
    try:
      data = plot_pipe_recv.recv()
      NumOfReg = data[0]
      minx = data[1]
      maxx = data[2]
      miny = data[3]
      maxy = data[4]
      Points_coordinates = data[5]
      frame = data[8]
      for i in range(NumOfReg):
      # For each region, plots the rectangle around the spot
      # and a cross at the center
        frame = cv2.rectangle(frame, (miny[i], minx[i]), (maxy[i] - 1,
                           maxx[i] - 1), (color, 0, 0), 1)
        frame = cv2.circle(frame, (int(Points_coordinates[i, 1]),
                           int(Points_coordinates[i, 0])), 1,
                           (255 - color, 0, 0), -1)
      cv2.imshow('frame', frame)
      cv2.waitKey(1)
    except KeyboardInterrupt:
      break
    except Exception as e:
      print "Exception in plotter:", e
      raise


def bary_wrapper(recv_):
  try:
    barycenter_opencv(recv_)
  except Exception as e:
    print "[barycenter process] Lost the spot:",e
    recv_.send("error")


def barycenter_opencv(recv_):
  """
  computation of the barycenter (moment 1 of image) on ZOI using OpenCV
  white_spot must be True if spots are white on a dark material
  The median filter helps a lot for real life images ...

  Args:
      recv_: Pipe which should receive data to calculate the barycenter of on
      or multiple spots.
              Received data is a tuple composed by:
                  - the picture where the spots are (numpy.ndarray)
                  - the min coordinate in x
                  - the min coordinate in y
                  - a boolean: True to update the threshold False otherwise.
                  - the current threshold
                  - border
                  - a boolean: true if the spot are white, false otherwise.
  """
  while True:
    image, minx, miny, update_tresh, thresh, NumOfReg, border, white_spot = recv_.recv()[:]
    # image1=sitk.GetImageFromArray(image)
    # sitk.WriteImage(image1,"img_videoExtenso%i.tiff"%os.getpid())
    if update_tresh:
      thresh = threshold_otsu(image)
    bw = cv2.medianBlur(image, 5) > thresh
    if not (white_spot):
      bw = 1 - bw
    M = cv2.moments(bw * 255.)

    if M['m00'] == 0:
      print "[barycenter process] Lost the spot!"
      recv_.send("error")
      return
    Px = M['m01'] / M['m00']
    Py = M['m10'] / M['m00']
    if NumOfReg == 1:
      a = M['mu20'] / M['m00']
      b = -M['mu11'] / M['m00']
      c = M['mu02'] / M['m00']
      # print "a,c : ", a,c
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
      Lx = max(np.abs(major_axis * np.cos(theta)), np.abs(minor_axis * np.sin(theta)))
      Ly = max(np.abs(major_axis * np.sin(theta)), np.abs(minor_axis * np.cos(theta)))
      # Px=Dx
      # Py=Dy

    # we add minx and miny to go back to global coordinate:
    Px += minx
    Py += miny
    miny_, minx_, h, w = cv2.boundingRect(
      (bw * 255).astype(np.uint8))  # cv2 returns x,y,w,h but x and y are inverted
    maxy_ = miny_ + h
    maxx_ = minx_ + w
    minx = minx - border + minx_
    miny = miny - border + miny_
    maxx = minx + border + maxx_
    maxy = miny + border + maxy_

    if NumOfReg == 1:
      recv_.send([Px, Py, minx, miny, maxx, maxy, Lx, Ly])
    else:
      recv_.send([Px, Py, minx, miny, maxx, maxy])


class VideoExtenso(MasterBlock):
  """
  Detects spots (1,2 or 4) on images, and evaluate the deformations Exx and Eyy.
  """

  def __init__(self,**kwargs):
    """
    Detects 1/2/4 spots, and evaluate the deformations Exx and Eyy.

    4 spots mode : deformations are evaluated on the distance between
                   centers of spots.
    2 spots mode : same, but deformation is only reliable on 1 axis.
    1 spot : deformation is evaluated on the major/minor axis of a theorical
             ellipse around the spot, projected over axis x and y.
             Results are less precise if your spot isn't big enough,
             but it is easier on smaller sample to only have 1 spot.

    Note that if this block lose the spots, it will play a song in the '/home/'
    repository. You need a .wav sound, python-pyglet and python-glob. This can
    be usefull if you have a long test to do, as the script doesn't stop when
    losing spots. Not to mention it is fun.

    Args:
        camera : string, {"Ximea","Jai"},default=Ximea
                See sensor.cameraSensor documentation.
        numdevice : int, default=0
                If you have multiple camera plugged, select the correct one.
        xoffset: int, default =0
                Offset on the x axis.
        yoffset: int, default =0
                Offset on the y axis.
        width: int, default = 2048
                Width of the image.
        height: int, default = 2048
                Height of the image.
        white_spot : Boolean, default=False
                Set to False if you have dark spots on a light surface.
        display : Boolean, default=True
                If True, a window will show up, displaying the spots and their
                computed position, userful to monitor the behavior
        update_tresh : Boolean, default=False
                Set to True if you want to re-evaluate the threshold for every
                new image.  Updside is that it allows you to follow more
                easily your spots even if your light changes. Downside is that
                it will change the area and possibly the shape of the spots,
                wich may inscrease the noise on the deformation and
                artificially change its value. This is especially true with a
                single spot configuration.
        labels : list of string, default = ['t(s)','Px','Py','Exx(%)','Eyy(%)']
                Labels of your output. Order is important.
        save_folder : str or None (default)
                If a path is definied, will save the images in this folder.
                If None, no saving

    Returns:
        dict :
                time : float
                        Time of the measure, relative to t0.
                Lx : float
                        Lenght (in pixels) of the spot.
                Ly : float
                        Width (in pixels) of the spot.
                Exx : float. Deformation = Lx/L0x
                Eyy : float. Deformation = Lxy/L0y
    """
    default_labels = ['t(s)', 'Px', 'Py', 'Exx(%)', 'Eyy(%)']
    for arg,default in [("camera","ximea"),
                        ("numdevice",0),
                        ('update_tresh',False),
                        ('security',False),
                        ('save_folder',None),
                        ('save_period',1), # Save one out of every N pictures
                        ('white_spot',False),
                        ('labels',default_labels),
                        ('display',True)]:
      setattr(self,arg,kwargs.get(arg,default))
      try:
        del kwargs[arg]
      except KeyError:
        pass
    self.camera_name = self.camera
    self.tve_kwargs = kwargs
    MasterBlock.__init__(self)
    # camera INIT with ZOI selection
    self.border = 4
    if self.save_folder is not None:
      if not os.path.exists(os.path.dirname(self.save_folder)):
        # check if the directory exists, otherwise create it
        os.makedirs(os.path.dirname(self.save_folder))

  def prepare(self):
    while True:
      # the following is to initialise the spot detection
      self.ve = tve(self.camera_name, self.numdevice,
                       {'white_spot': self.white_spot,
                        'border': self.border},**self.tve_kwargs)
      for attr in ['minx','maxx','miny','maxy','NumOfReg','L0x','L0y',
                   'thresh','Points_coordinates']:#,'width','height',
                   #'exposure','gain']:
        setattr(self,attr,self.ve.videoextenso[attr])
      if self.NumOfReg == 4 or self.NumOfReg == 2 or self.NumOfReg == 1:
        break
      else:  # If detection goes wrong, start again, may not be usefull now ?
        print " Spots detected : ", self.NumOfReg

    #self.ve.sensor.new(self.exposure, self.width, self.height,
    #                       self.xoffset, self.yoffset, self.gain)
    #image = self.ve.sensor.get_image()
    # eliminate the first frame, most likely corrupted
    self.proc_bary = []
    self.recv_ = []
    self.send_ = []
    for i in range(self.NumOfReg):
      # Creating the spot processes and their pipe
      _r,_s = Pipe()
      self.recv_.append(_r)
      self.send_.append(_s)

      self.proc_bary.append(Process(
                              target=bary_wrapper, args=(self.recv_[i],)))
      self.proc_bary[i].start()
    if self.display:
      # Creating displayer process
      self.plot_pipe_recv, self.plot_pipe_send = Pipe()
      Process(target=plotter, args=(self.plot_pipe_recv,)).start()

  def main(self):
    j = 0
    last_ttimer = time.time()

    while True:
      try:
        t2 = time.time()
        t_image,image = self.ve.sensor.read_image()  # read a frame
        if self.save_folder is not None and not j%self.save_period:
          image1 = sitk.GetImageFromArray(image)
          sitk.WriteImage(image1,
                    self.save_folder + "img_videoExtenso%.5d_%4.4f.tiff" % (
                                                        j,t2-self.t0))

        # for each spot, calulate the news coordinates of the center, based on
        # previous coordinate and border.
        for i in range(self.NumOfReg):
          self.send_[i].send([image[int(self.minx[i]) - 1:int(self.maxx[i]) + 1,
                         int(self.miny[i]) - 1:int(self.maxy[i]) + 1],
                         self.minx[i] - 1, self.miny[i] - 1,
                         self.update_tresh, self.thresh, self.NumOfReg,
                         self.border, self.white_spot])
        if self.NumOfReg == 1:
          self.Points_coordinates[i, 0], self.Points_coordinates[i, 1],\
          self.minx[i], self.miny[i], \
          self.maxx[i], self.maxy[i], Lx, Ly = self.send_[i].recv()[:]
        else:
          for i in range(self.NumOfReg):
            self.Points_coordinates[i, 0], self.Points_coordinates[i, 1],\
             self.minx[i], self.miny[i], \
            self.maxx[i], self.maxy[i] = self.send_[i].recv()[:]
            # self.minx[i],self.miny[i],self.maxx[i],self.maxy[i]
        minx_ = self.minx.min()
        miny_ = self.miny.min()
        maxx_ = self.maxx.max()
        maxy_ = self.maxy.max()
        if self.NumOfReg == 4 or self.NumOfReg == 2:
          Lx = self.Points_coordinates[:, 0].max() - \
                              self.Points_coordinates[:, 0].min()
          Ly = self.Points_coordinates[:, 1].max() - \
                              self.Points_coordinates[:, 1].min()
          Dx = 100. * ((Lx) / self.L0x - 1.)
          Dy = 100. * ((Ly) / self.L0y - 1.)
        elif self.NumOfReg == 1:
          Dy = 100. * ((Ly) / self.L0y - 1.)
          Dx = 100. * ((Lx) / self.L0x - 1.)
        array = [t_image - self.t0, str(self.Points_coordinates[:, 0]),
                     str(self.Points_coordinates[:, 1]), Dx, Dy]
        self.send(array)
        self.Points_coordinates[:, 1] -= miny_  # go back to local repere
        self.Points_coordinates[:, 0] -= minx_
        if self.display and not j % 80:
          # every 80 round, send an image to the plot function below,
          # that displays the cropped image,
          # LX, Ly and the position of the area around the spots
          self.plot_pipe_send.send(
            [self.NumOfReg, self.minx - minx_,
             self.maxx - minx_, self.miny - miny_, self.maxy - miny_,
             self.Points_coordinates, self.L0x, self.L0y,
             image[minx_:maxx_, miny_:maxy_],
             self.white_spot])

        if j % 100 == 0 and j > 0:
          t_now = time.time()
          stdout.write("\rFPS: %2.2f" % (100 / (t_now - last_ttimer)))
          stdout.flush()
          last_ttimer = t_now
        j += 1
      except ValueError:  # if lost spots in barycenter
        print "[VideoExtenso] SPOTS LOST, abort mission!"
        try:
          import pyglet
          import glob
          import random
          song_list = glob.glob('/home/*.wav')
          song = pyglet.media.load(random.choice(song_list))
          song.play()
          pyglet.clock.schedule_once(lambda x: pyglet.app.exit(), 10)
          # stop music after 10 sec
          pyglet.app.run()
        except Exception as e:
          print "No music because : ", e
        if self.security:
          print 'Exception Video Extenso Security'
          try:
            for output in self.outputs:
              output.send("error")
          except TimeoutError:
            raise
          except AttributeError:  # if no outputs
            pass
        raise Exception("Spots lost")
      except KeyboardInterrupt:
        self.ve.sensor.close()
        print 'KeyboardInterrupt\n'
        break
      except Exception as e:
        print "Exception in videoextenso : ", e
        for i in range(0, self.NumOfReg):
          self.proc_bary[i].terminate()
        raise
