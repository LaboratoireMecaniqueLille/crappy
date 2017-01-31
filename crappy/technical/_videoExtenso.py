# coding: utf-8
##  @addtogroup technical
# @{

##  @defgroup videoExtenso videoExtenso
# @{

## @file _videoExtenso.py
# @brief Opens a camera device and prepares viedoextenso
#
# @author Victor Couty
# @version 0.1
# @date 26/01/2017

from __future__ import print_function


from . import get_camera_config
from ._camera import TechnicalCamera

class TechnicalVideoExtenso(TechnicalCamera):
  """
  Opens a camera device and initialise it.
  """

  def __init__(self, camera="Ximea", num_device=0, videoextenso={},**kwargs):
    """
This class is like technical camera but it includes all the 
necessary parameters for videoextenso
    """
    TechnicalCamera.__init__(self, camera=camera, num_device=num_device, 
                                        config=False, **kwargs)
    self.videoextenso = videoextenso
    data = get_camera_config(self.sensor,videoextenso)
    for i,p in enumerate(['minx','maxx','miny','maxy',
                    'NumOfReg', 'L0x','L0y','thresh','Points_coordinates']):
      self.videoextenso[p] = data[i]
