# coding: utf-8

import crappy
import cv2


class Ellipse(crappy.tool.camera_config.Overlay):

  def __init__(self, center_x, center_y, x_axis, y_axis):

    super().__init__()

    self._center_x = center_x
    self._center_y = center_y
    self._x_axis = x_axis
    self._y_axis = y_axis

  def draw(self, img):

    thickness = max(img.shape[0] // 480, img.shape[1] // 640, 1) + 1
    cv2.ellipse(img,
                (self._center_x, self._center_y),
                (self._x_axis, self._y_axis),
                0,
                0,
                360,
                0,
                thickness)


class CustomCameraProcess(crappy.blocks.camera_processes.CameraProcess):

  def __init__(self,
               scale_factor=1.2,
               min_neighbors=3):

    super().__init__()

    self._eye_cascade = None

    self._scale_factor = scale_factor
    self._min_neighbors = min_neighbors

  def init(self):

    self._eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')

  def loop(self):

    eyes = self._eye_cascade.detectMultiScale(self.img,
                                              scaleFactor=self._scale_factor,
                                              minNeighbors=self._min_neighbors)
    to_draw = list()
    for (x, y, width, height) in eyes:
      to_draw.append(Ellipse(int(x + width / 2), int(y + height / 2),
                             int(width / 2), int(height / 2)))

    self.send_to_draw(to_draw)
    self.send({'t(s)': self.metadata['t(s)'], 'eyes': eyes})


class CustomCameraBlock(crappy.blocks.Camera):

  def __init__(self,
               camera,
               transform=None,
               config=True,
               display_images=False,
               displayer_backend=None,
               displayer_framerate=5,
               software_trig_label=None,
               display_freq=False,
               freq=200,
               debug=False,
               save_images=False,
               img_extension="tiff",
               save_folder=None,
               save_period=1,
               save_backend=None,
               image_generator=None,
               img_shape=None,
               img_dtype=None,
               scale_factor=1.2,
               min_neighbors=3,
               **kwargs):

    super().__init__(camera=camera,
                     transform=transform,
                     config=config,
                     display_images=display_images,
                     displayer_backend=displayer_backend,
                     displayer_framerate=displayer_framerate,
                     software_trig_label=software_trig_label,
                     display_freq=display_freq,
                     freq=freq,
                     debug=debug,
                     save_images=save_images,
                     img_extension=img_extension,
                     save_folder=save_folder,
                     save_period=save_period,
                     save_backend=save_backend,
                     image_generator=image_generator,
                     img_shape=img_shape,
                     img_dtype=img_dtype,
                     **kwargs)

    self._scale_factor = scale_factor
    self._min_neighbors = min_neighbors

  def prepare(self):

    self.process_proc = CustomCameraProcess(self._scale_factor,
                                            self._min_neighbors)
    super().prepare()


if __name__ == '__main__':

  cam = CustomCameraBlock('Webcam',
                          display_images=True,
                          save_images=False,
                          freq=20,
                          displayer_framerate=20,
                          scale_factor=1.2,
                          min_neighbors=6)

  stop = crappy.blocks.StopButton()

  reader = crappy.blocks.LinkReader()

  crappy.link(cam, reader)

  crappy.start()
