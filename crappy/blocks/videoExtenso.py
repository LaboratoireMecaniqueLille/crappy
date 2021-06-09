# coding: utf-8

from ..tool.videoextenso import LostSpotError, Video_extenso as Ve
from ..tool.videoextensoConfig import VE_config
from .camera import Camera
from .._global import OptionalModule

try:
  import cv2
except (ModuleNotFoundError, ImportError):
  cv2 = OptionalModule("opencv-python")


class Video_extenso(Camera):
  """
  Measure the deformation for the video of dots on the sample.

  Warning!
    This requires the user to select the ROI to make the spot detection.

    Once done, it will return the deformation (in %) along X and Y axis.

  Note:
    It also returns a list of tuples, which are the coordinates (in pixel)
    of the barycenter of the spots.

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

  def __init__(self,
               camera,
               save_folder=None,
               verbose=False,
               labels=None,
               fps_label=False,
               img_name="{self.loops:06d}_{t-self.t0:.6f}",
               ext='tiff',
               save_period=1,
               save_backend=None,
               transform=None,
               input_label=None,
               config=False,
               show_image=False,
               wait_l0=False,
               end=True,
               white_spots=False,
               update_thresh=False,
               num_spots="auto",
               safe_mode=False,
               border=5,
               min_area=150,
               blur=5,
               **kwargs):
    cam_kw = {}
    self.niceness = -5
    # Kwargs to be given to the camera BLOCK
    # ie save_folder, config, etc... but NOT the labels

    cam_kw['save_folder'] = save_folder
    cam_kw['verbose'] = verbose
    cam_kw['fps_label'] = fps_label
    cam_kw['img_name'] = img_name
    cam_kw['ext'] = ext
    cam_kw['save_period'] = save_period
    cam_kw['save_backend'] = save_backend
    cam_kw['transform'] = transform
    cam_kw['input_label'] = input_label
    cam_kw['config'] = config

    cam_kw['config'] = False  # VE has its own config window
    self.verbose = cam_kw['verbose']  # Also, we keep the verbose flag

    self.labels = ['t(s)', 'Coord(px)', 'Eyy(%)', 'Exx(%)'] \
      if labels is None else labels
    self.show_image = show_image
    self.wait_l0 = wait_l0
    self.end = end

    self.ve_kwargs = dict()
    self.ve_kwargs['white_spots'] = white_spots
    self.ve_kwargs['update_thresh'] = update_thresh
    self.ve_kwargs['num_spots'] = num_spots
    self.ve_kwargs['safe_mode'] = safe_mode
    self.ve_kwargs['border'] = border
    self.ve_kwargs['min_area'] = min_area
    self.ve_kwargs['blur'] = blur

    self.cam_kw = kwargs
    self.cam_kw.update(cam_kw)
    self.cam_kw['labels'] = self.labels
    Camera.__init__(self, camera, **self.cam_kw)

  def prepare(self, **_):
    Camera.prepare(self, send_img=False)
    self.ve = Ve(**self.ve_kwargs)
    config = VE_config(self.camera, self.ve)
    config.main()
    self.ve.start_tracking()
    if self.show_image:
      try:
        flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
      except AttributeError:
        flags = cv2.WINDOW_NORMAL
      cv2.namedWindow("Videoextenso", flags)
    self.loops = 0
    self.last_fps_print = 0
    self.last_fps_loops = 0

  def loop(self):
    t, img = self.get_img()
    if self.inputs and not self.input_label and self.inputs[0].poll():
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
      for miny, minx, maxy, maxx in boxes:
        img[miny:miny+1, minx:maxx] = 255
        img[maxy:maxy+1, minx:maxx] = 255
        img[miny:maxy, minx:minx+1] = 255
        img[miny:maxy, maxx:maxx+1] = 255
      cv2.imshow("Videoextenso", img)
      cv2.waitKey(5)

    centers = [(r['y'], r['x']) for r in self.ve.spot_list]
    if not self.wait_l0:
      self.send([t-self.t0, centers]+d)
    else:
      self.send([t - self.t0, [(0, 0)] * 4, 0, 0])

  def lost_loop(self):
    t, img = self.get_img()
    if self.show_image:
      cv2.imshow("Videoextenso", img)
      cv2.waitKey(5)

  def finish(self):
    self.ve.stop_tracking()
    if self.show_image:
      cv2.destroyAllWindows()
    Camera.finish(self)
