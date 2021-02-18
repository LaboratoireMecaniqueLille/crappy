# coding: utf-8

import cv2

from ..tool.videoextenso import LostSpotError,Video_extenso as VE
from ..tool.videoextensoConfig import VE_config
from .camera import Camera,kw as default_cam_block_kw


class Video_extenso(Camera):
  """
  Measure the deformation for the video of dots on the sample.

  Warning!
    This requires the user to select the ROI to make the spot detection.

    Once done, it will return the deformation (in %) along X and Y axis.

  Note:
    It also returns a list of tuples, which are the coordinates (in pixel)
    of the barycenters of the spots.

    Optionally, it can save images.

    The initial length is reset when receiving data from a parent block.

  Args:
    - camera (str, mandatory): The name of the camera class to use.
    - labels (list, default: ['t(s)', 'Coord(px)', 'Eyy(%)', 'Exx(%)']): The
      labels of the output.
    - wait_l0: If set to True, the block send only zeros until the initial
      length is reset by receiving data from an input.
    - end (bool, default: True): If True, the block will stop the Crappy
      program when the spots are lost, else it will just stop sending data.

  """
  def __init__(self,camera,**kwargs):
    self.niceness = -5
    cam_kw = {}
    # Kwargs to be given to the camera BLOCK
    # ie save_folder, config, etc... but NOT the labels
    for k,v in default_cam_block_kw.items():
      if k == 'labels':
        continue
      cam_kw[k] = kwargs.pop(k,v)
    cam_kw['config'] = False # VE has its own config window
    self.verbose = cam_kw['verbose'] # Also, we keep the verbose flag
    Camera.__init__(self,camera,**cam_kw)
    default_labels = ['t(s)', 'Coord(px)', 'Eyy(%)', 'Exx(%)']
    for arg,default in [("labels",default_labels),
                        ("show_image",False),
                        ("wait_l0",False),
                        ("end",True),
                        ]:
      try:
        setattr(self,arg,kwargs[arg])
        del kwargs[arg]
      except KeyError:
        setattr(self,arg,default)
    self.ve_kwargs = {}
    for arg in ['white_spots','update_thresh','num_spots',
        'safe_mode','border','min_area']:
      if arg in kwargs:
        self.ve_kwargs[arg] = kwargs[arg]
        del kwargs[arg]
    self.cam_kwargs = kwargs

  def prepare(self):
    Camera.prepare(self)
    self.ve = VE(**self.ve_kwargs)
    config = VE_config(self.camera,self.ve)
    config.main()
    self.ve.start_tracking()
    if self.show_image:
      try:
        flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
      except AttributeError:
        flags = cv2.WINDOW_NORMAL
      cv2.namedWindow("Videoextenso",flags)
    self.loops = 0
    self.last_fps_print = 0
    self.last_fps_loops = 0

  def loop(self):
    t,img = self.get_img()
    if self.inputs and self.inputs[0].poll():
      self.inputs[0].clear()
      self.wait_l0 = False
      print("[VE block] resetting L0")
      self.ve.save_length()
    try:
      d = self.ve.get_def(img)
    except LostSpotError:
      print("[VE block] Lost spots, terminating")
      self.ve.stop_tracking()
      if self.end:
        raise
      else:
        self.loop = self.lost_loop
        return
    if self.show_image:
      boxes = [r['bbox'] for r in self.ve.spot_list]
      for miny,minx,maxy,maxx in boxes:
        img[miny:miny+1,minx:maxx] = 255
        img[maxy:maxy+1,minx:maxx] = 255
        img[miny:maxy,minx:minx+1] = 255
        img[miny:maxy,maxx:maxx+1] = 255
      cv2.imshow("Videoextenso",img)
      cv2.waitKey(5)

    centers = [(r['y'],r['x']) for r in self.ve.spot_list]
    if not self.wait_l0:
      self.send([t-self.t0,centers]+d)
    else:
      self.send([t-self.t0,[(0,0)]*4,0,0])

  def lost_loop(self):
    t,img = self.get_img()
    if self.show_image:
      cv2.imshow("Videoextenso",img)
      cv2.waitKey(5)

  def finish(self):
    self.ve.stop_tracking()
    if self.show_image:
      cv2.destroyAllWindows()
    Camera.finish(self)
