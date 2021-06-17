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
  """Measures the deformation from video by tracking dots on the sample.

  This requires the user to select the ROI to make the spot detection. Once
  done, it will return the deformation (in `%`) along `X` and `Y` axis.
  Optionally, it can save images.

  It also returns a :obj:`list` of :obj`tuple`, which are the coordinates (in
  pixel) of the barycenter of the spots.

  The initial length is reset when receiving data from a parent block.
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
    """Sets the args and initializes the camera.

    Args:
      camera (:obj:`str`): The name of the camera to control. See
        :ref:`Cameras` for an exhaustive list of available ones.
      save_folder (:obj:`str`, optional): The directory to save images to. If
        it doesn't exist it will be created. If :obj:`None` the images won't be
        saved.
      verbose (:obj:`bool`, optional): If :obj:`True`, the block will print the
        number of `loops/s`.
      labels (:obj:`list`, optional): Names of the labels for the output.
      fps_label (:obj:`str`, optional): If set, ``self.max_fps`` will be set to
        the value received by the block with this label.
      img_name (:obj:`str`, optional): Template for the name of the image to
        save. It is evaluated as an `f-string`.
      ext (:obj:`str`, optional): Extension of the image. Make sure it is
        supported by the saving backend.
      save_period (:obj:`int`, optional): Will save only one in `x` images.
      save_backend (:obj:`str`, optional): Module to use to save the images.
        The supported backends are: :mod:`sitk` (SimpleITK), :mod:`cv2`
        (OpenCV) and :mod:`pil` (Pillow). If :obj:`None`, will try :mod:`sitk`
        and then :mod:`cv2` if not successful.
      transform (:obj:`function`, optional): Function to be applied on the
        image before sending. It will not be applied on the saved images.
      input_label (:obj:`str`, optional): If specified, the image will not be
        read from a camera object but from this label.
      config (:obj:`bool`, optional): Show the popup for config ?
      show_image:
      wait_l0 (:obj:`bool`, optional): If set to :obj:`True`, the block sends
        only zeros until the initial length is reset by receiving data from an
        input.
      end (:obj:`bool`, optional): If :obj:`True`, the block will stop the
        Crappy program when the spots are lost, else it will just stop sending
        data.
      white_spots: Set to :obj:`True` if the spots are lighter than the
        surroundings, else set to :obj:`False`.
      update_thresh: Should the threshold be updated in each round ? If so
        there are lower chances to lose the spots but there will be more noise
        in the measurement.
      num_spots: The number of spots to detect. Helps for spot detection and
        allows to force detection of a given number of spots (`"auto"` works
        fine most of the time). Can be set to:
        ::

          "auto", 2, 3, 4

      safe_mode: If set to :obj:`False`, it will try hard to catch the spots
        when losing them. Could result in incoherent values without crash. Set
        to :obj:`True` when security is a concern.
      border: The number of pixels that will be added to the limits of the
        boundingbox.
      min_area: Filters regions with an area smaller than this value among the
        selected regions.
      blur: Median blur to be added to the image to smooth out irregularities
        and make detection more reliable.
      **kwargs: Any additional specific argument to pass to the camera.
    """

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

  def prepare(self):
    Camera.prepare(self, send_img=False)
    self.ve = Ve(**self.ve_kwargs)
    config = VE_config(self.camera, self.ve)
    config.main()
    if not self.ve.spot_list:
      print("No markers were detected for videoextenso! "
          "Please select the markers and make sure they are detected. "
          "A box should appear around the markers when they are detected. "
          "If not, make sure the white_spots argument is correctly set "
          "and the markers are large enough.")
      raise AttributeError("Missing VE Markers")
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
        img[miny:miny + 1, minx:maxx] = 255
        img[maxy:maxy + 1, minx:maxx] = 255
        img[miny:maxy, minx:minx + 1] = 255
        img[miny:maxy, maxx:maxx + 1] = 255
      cv2.imshow("Videoextenso", img)
      cv2.waitKey(5)

    centers = [(r['y'], r['x']) for r in self.ve.spot_list]
    if not self.wait_l0:
      self.send([t - self.t0, centers] + d)
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
