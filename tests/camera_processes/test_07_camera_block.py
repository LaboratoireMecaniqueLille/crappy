# coding: utf-8

from multiprocessing import Event, Value
from threading import BrokenBarrierError
from typing import Any
import logging
import numpy as np
from crappy import Block
from crappy._global import CameraPrepareError, CameraRuntimeError
from crappy.blocks.camera import Camera
import crappy.blocks.camera as camera_module

from .camera_process_test_base import (CameraProcessTestBase,
                                       TestCameraProcess, TestLink)


class TrackingCameraProcess(TestCameraProcess):
  """CameraProcess stand-in recording how the Camera Block manages it."""

  def __init__(self, *args, **kwargs) -> None:
    """Initializes tracking Events and constructor argument storage."""

    super().__init__()

    self.started = Event()
    self.terminated = Event()
    self.shared = Event()
    self._alive = False
    self.args = args
    self.kwargs = kwargs

  def start(self) -> None:
    """Records that the process would have been started."""

    self.started.set()

  def is_alive(self) -> bool:
    """Returns the fake liveness state."""

    return self._alive

  def terminate(self) -> None:
    """Records that the Camera Block asked the process to terminate."""

    self.terminated.set()
    self._alive = False

  def set_shared(self, *args, **kwargs) -> None:
    """Records shared object injection and delegates to the parent method."""

    super().set_shared(*args, **kwargs)
    self.shared.set()


class TrackingImageSaver(TrackingCameraProcess):
  """ImageSaver stand-in recording constructor arguments."""

  def __init__(self,
               img_extension: str = "tiff",
               save_folder: str | None = None,
               save_period: int = 1,
               save_backend: str | None = None,
               send_msg: bool = False) -> None:
    """Records ImageSaver-specific constructor arguments."""

    super().__init__(img_extension=img_extension,
                     save_folder=save_folder,
                     save_period=save_period,
                     save_backend=save_backend,
                     send_msg=send_msg)

    self.img_extension = img_extension
    self.save_folder = save_folder
    self.save_period = save_period
    self.save_backend = save_backend
    self.send_msg = send_msg


class TrackingDisplayer(TrackingCameraProcess):
  """Displayer stand-in recording constructor arguments."""

  def __init__(self,
               title: str,
               framerate: float,
               backend: str | None = None) -> None:
    """Records Displayer-specific constructor arguments."""

    super().__init__(title=title, framerate=framerate, backend=backend)

    self.title = title
    self.framerate = framerate
    self.backend = backend


class CameraBlockTestBase(CameraProcessTestBase):
  """Base test class for Camera Block interaction tests."""

  def __init__(self, *args, **kwargs) -> None:
    """Initializes the parent test case and tracked Camera Block."""

    super().__init__(*args, **kwargs)

    self._camera_block: Camera | None = None
    self._original_image_saver = camera_module.ImageSaver
    self._original_displayer = camera_module.Displayer

  def setUp(self) -> None:
    """Replaces concrete CameraProcess classes with stand-ins."""

    camera_module.ImageSaver = TrackingImageSaver
    camera_module.Displayer = TrackingDisplayer

  def tearDown(self) -> None:
    """Restores patched classes and releases Camera Block state."""

    camera_module.ImageSaver = self._original_image_saver
    camera_module.Displayer = self._original_displayer

    try:
      if self._camera_block is not None:
        if self._camera_block._manager is not None:
          try:
            self._camera_block.finish()
          except (BrokenBarrierError, FileNotFoundError, AttributeError):
            pass

    finally:
      Camera.cam_count.clear()
      Block.reset()
      super().tearDown()

  def make_camera(self, **kwargs) -> Camera:
    """Instantiates a Camera Block using an image generator."""

    image = np.arange(20, dtype=np.uint8).reshape(4, 5)

    def generator(_, __) -> np.ndarray:
      """Returns a deterministic frame for Camera.loop tests."""

      return image

    defaults: dict[str, Any] = dict(camera='unused',
                                    config=False,
                                    image_generator=generator,
                                    img_shape=image.shape,
                                    img_dtype='uint8')
    defaults.update(kwargs)

    self._camera_block = Camera(**defaults)
    self._camera_block._log_level = logging.CRITICAL
    self._camera_block._instance_t0 = Value('d', 1.0)

    return self._camera_block


class TestCameraBlock(CameraBlockTestBase):
  """Tests the contract between CameraProcess and the Camera Block."""

  def test_prepare_process_proc(self) -> None:
    """Tests how Camera.prepare shares objects with a processing process."""

    process = TrackingCameraProcess()
    camera = self.make_camera()
    camera.process_proc = process
    camera.labels = ['x', 'y']
    camera.outputs = [TestLink()]

    camera.prepare()

    self.assertTrue(process.shared.is_set())
    self.assertTrue(process.started.is_set())
    self.assertEqual(camera._cam_barrier.parties, 2)

    self.assertIs(process._img_array, camera._img_array)
    self.assertIs(process._data_dict, camera._metadata)
    self.assertIs(process._lock, camera._proc_lock)
    self.assertIs(process._cam_barrier, camera._cam_barrier)
    self.assertIs(process._stop_event, camera._stop_event_cam)
    self.assertIsNone(process._to_draw_conn)
    self.assertIs(process._outputs, camera.outputs)
    self.assertIs(process._labels, camera.labels)
    self.assertIs(process._log_queue, camera._log_queue)
    self.assertEqual(process._log_level, camera._log_level)
    self.assertEqual(process._shape, camera._img_shape)
    self.assertEqual(process._dtype, np.dtype(camera._img_dtype))

  def test_prepare_process_proc_with_display(self) -> None:
    """Tests overlay pipe sharing between processing and display processes."""

    process = TrackingCameraProcess()
    camera = self.make_camera(display_images=True,
                              displayer_backend='cv2',
                              displayer_framerate=20)
    camera.process_proc = process

    camera.prepare()

    self.assertIsInstance(camera._display_proc, TrackingDisplayer)
    self.assertTrue(process.shared.is_set())
    self.assertTrue(camera._display_proc.shared.is_set())
    self.assertEqual(camera._cam_barrier.parties, 3)

    self.assertIs(process._to_draw_conn, camera._overlay_conn_in)
    self.assertIs(camera._display_proc._to_draw_conn,
                  camera._overlay_conn_out)
    self.assertIs(camera._display_proc._lock, camera._disp_lock)
    self.assertEqual(camera._display_proc._outputs, list())
    self.assertEqual(camera._display_proc._labels, list())
    self.assertEqual(camera._display_proc.backend, 'cv2')
    self.assertEqual(camera._display_proc.framerate, 20)

  def test_prepare_image_saver(self) -> None:
    """Tests how Camera.prepare shares objects with the image saver."""

    camera = self.make_camera(save_images=True,
                              save_period=3,
                              save_backend='npy',
                              img_extension='npy')
    camera.outputs = [TestLink()]

    camera.prepare()

    self.assertIsInstance(camera._save_proc, TrackingImageSaver)
    self.assertTrue(camera._save_proc.shared.is_set())
    self.assertTrue(camera._save_proc.started.is_set())
    self.assertEqual(camera._cam_barrier.parties, 2)

    self.assertTrue(camera._save_proc.send_msg)
    self.assertEqual(camera._save_proc.save_period, 3)
    self.assertEqual(camera._save_proc.save_backend, 'npy')

    self.assertIs(camera._save_proc._img_array, camera._img_array)
    self.assertIs(camera._save_proc._data_dict, camera._metadata)
    self.assertIs(camera._save_proc._lock, camera._save_lock)
    self.assertIs(camera._save_proc._cam_barrier, camera._cam_barrier)
    self.assertIs(camera._save_proc._stop_event, camera._stop_event_cam)
    self.assertIsNone(camera._save_proc._to_draw_conn)
    self.assertIs(camera._save_proc._outputs, camera.outputs)
    self.assertEqual(camera._save_proc._labels, list())

  def test_prepare_image_saver_with_process_proc(self) -> None:
    """Tests image saver messages when a processing process exists."""

    process = TrackingCameraProcess()
    camera = self.make_camera(save_images=True)
    camera.process_proc = process
    camera.outputs = [TestLink()]

    camera.prepare()

    self.assertIsInstance(camera._save_proc, TrackingImageSaver)
    self.assertFalse(camera._save_proc.send_msg)
    self.assertEqual(camera._cam_barrier.parties, 3)
    self.assertIs(camera._save_proc._outputs, camera.outputs)
    self.assertIs(process._outputs, camera.outputs)

  def test_begin(self) -> None:
    """Tests that Camera.begin releases a ready Camera barrier."""

    camera = self.make_camera()
    camera.prepare()

    camera.begin()

    self.assertFalse(camera._cam_barrier.broken)

  def test_begin_broken_barrier(self) -> None:
    """Tests that Camera.begin converts a broken barrier into an error."""

    camera = self.make_camera()
    camera.prepare()
    camera._cam_barrier.abort()

    with self.assertRaises(CameraPrepareError):
      camera.begin()

  def test_loop_writes_shared_frame(self) -> None:
    """Tests that Camera.loop writes frames for CameraProcess instances."""

    camera = self.make_camera()
    camera.prepare()

    camera.loop()

    self.assertEqual(camera._metadata['ImageUniqueID'], 0)
    self.assertIn('DateTimeOriginal', camera._metadata)
    self.assertIn('SubsecTimeOriginal', camera._metadata)

    expected = np.arange(20, dtype=np.uint8).reshape(4, 5)
    np.testing.assert_array_equal(camera._img, expected)
    self.assertEqual(camera._loop_count, 1)

  def test_loop_child_stop_event(self) -> None:
    """Tests that Camera.loop fails if a CameraProcess requested a stop."""

    camera = self.make_camera()
    camera.prepare()
    camera._stop_event_cam.set()

    with self.assertRaises(CameraRuntimeError):
      camera.loop()

  def test_finish_terminates_live_processes(self) -> None:
    """Tests that Camera.finish stops every managed CameraProcess."""

    process = TrackingCameraProcess()
    camera = self.make_camera(save_images=True, display_images=True)
    camera.process_proc = process

    camera.prepare()
    process._alive = True
    camera._save_proc._alive = True
    camera._display_proc._alive = True

    camera.finish()

    self.assertTrue(camera._stop_event_cam.is_set())
    self.assertTrue(process.terminated.is_set())
    self.assertTrue(camera._save_proc.terminated.is_set())
    self.assertTrue(camera._display_proc.terminated.is_set())
    self.assertFalse(process.is_alive())
    self.assertFalse(camera._save_proc.is_alive())
    self.assertFalse(camera._display_proc.is_alive())
