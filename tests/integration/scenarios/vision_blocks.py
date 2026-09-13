# coding: utf-8

import json
from pathlib import Path
from time import time

import numpy as np

import crappy
from crappy.blocks import Block
from crappy.blocks.vision import (CameraSource, DICVEProcessor,
                                  DISCorrelProcessor, ImageRecorder,
                                  VideoExtensoProcessor, VisionBlock)
from crappy.blocks.vision.block import ConfigRequest
from crappy.tool.camera_config import CameraConfig, SpotsBoxes


def generate_vision_test_image(_: float, __: float) -> np.ndarray:
  """Returns a deterministic greyscale image for VisionBlock scenarios."""

  y, x = np.indices((48, 64))
  return ((3 * x + 5 * y) % 256).astype(np.uint8)


def _make_vision_speckle() -> np.ndarray:
  """Returns a reproducible textured image for correlation scenarios."""

  rng = np.random.default_rng(12345)
  return rng.integers(0, 256, size=(128, 128), dtype=np.uint8)


def _make_strain_generator() -> crappy.blocks.Generator:
  """Returns the finite strain command shared by correlation scenarios."""

  return crappy.blocks.Generator(
    ({'type': 'Ramp',
      'speed': 6,
      'condition': 'delay=1.0',
      'init_value': 0},),
    cmd_label='Exx(%)',
    spam=True,
    end_delay=0.2,
    freq=20)


def _make_strain_camera() -> CameraSource:
  """Returns a headless generated CameraSource for correlation scenarios."""

  return CameraSource(
    '',
    image_generator=crappy.tool.ApplyStrainToImage(_make_vision_speckle()),
    config=False,
    allow_downstream_config=False,
    img_shape=(128, 128),
    img_dtype='uint8',
    freq=30)


class FiniteConfiguredImageSource(VisionBlock):
  """Publishes finite frames and answers downstream configuration requests."""

  _shape = (12, 16)
  _dtype = 'uint16'

  def __init__(self,
               artifact_path: Path,
               frame_count: int = 12) -> None:
    """Stores output settings and the source-side artifact destination."""

    super().__init__(img_shape=self._shape,
                     img_dtype=self._dtype,
                     freq=30,
                     debug=False)
    self.name = 'Configured image source'
    self._artifact_path = artifact_path
    self._frame_count = frame_count
    self._sequence = 0

  def prepare(self) -> None:
    """Answers every request before allocating the shared image buffer."""

    answered_tokens = list()
    for request in self.config_requests_in:
      token = request.kwargs['token']
      self.send_config(request, (token, self._shape, self._dtype))
      answered_tokens.append(token)

    super().prepare()

    with self._artifact_path.open('w', encoding='utf-8') as file:
      json.dump({'answered_tokens': answered_tokens,
                 'memory_name': self._out_link_data.memory_name,
                 'shape': self._shape,
                 'dtype': self._dtype}, file)

  def loop(self) -> None:
    """Publishes a coherent constant-valued frame and eventually stops."""

    sequence = self._sequence
    image = np.full(self._shape, sequence, dtype=self._dtype)
    metadata = {'t(s)': time() - self.t0,
                'ImageUniqueID': sequence,
                'sequence': sequence,
                'source': self.name}
    self.send_img(metadata, image)
    self._sequence += 1

    if self._sequence >= self._frame_count:
      self.stop()


class ConfiguredFrameProbe(VisionBlock):
  """Requests source configuration and records received image observations."""

  def __init__(self,
               artifact_path: Path,
               token: str,
               name: str) -> None:
    """Stores the request token, stable name, and artifact destination."""

    super().__init__(freq=100, debug=False)
    self.name = name
    self._artifact_path = artifact_path
    self._token = token
    self._configs: dict[str, tuple | None] = dict()
    self._observations: list[dict] = list()

  def request_config(self, source: str) -> ConfigRequest:
    """Requests a source-specific token over a required configuration Pipe."""

    return ConfigRequest(requester=self.name,
                         args=tuple(),
                         kwargs={'token': self._token},
                         configurator=CameraConfig,
                         img_source=source,
                         required=True)

  def prepare(self) -> None:
    """Receives configuration before attaching to the image buffer."""

    self._configs = self.recv_configs()
    super().prepare()

  def loop(self) -> None:
    """Records JSON-safe evidence for each newest frame received."""

    for link_name in self.receive_imgs():
      received = self.last_received[link_name]
      if received.metadata is None:
        raise RuntimeError("Received image metadata should not be empty")

      metadata = received.metadata
      image = received.img
      self._observations.append({
        'transport_id': received.id,
        'image_id': metadata['ImageUniqueID'],
        'sequence': metadata['sequence'],
        'timestamp': metadata['t(s)'],
        'shape': image.shape,
        'dtype': str(image.dtype),
        'first_pixel': int(image.flat[0]),
        'checksum': int(image.sum()),
      })

  def finish(self) -> None:
    """Writes observations and always releases the attached shared memory."""

    try:
      with self._artifact_path.open('w', encoding='utf-8') as file:
        json.dump({'configs': self._configs,
                   'observations': self._observations}, file)
    finally:
      super().finish()


class FiniteSpotImageSource(VisionBlock):
  """Publishes finite spot images and configures a VideoExtensoProcessor."""

  _shape = (96, 96)
  _spot_declarations = ((18, 18, 12, 12), (60, 60, 12, 12))

  def __init__(self,
               artifact_path: Path,
               frame_count: int = 20) -> None:
    """Builds the static spot image and stores scenario settings."""

    super().__init__(img_shape=self._shape,
                     img_dtype='uint8',
                     freq=20,
                     debug=False)
    self.name = 'Configured spot image source'
    self._artifact_path = artifact_path
    self._frame_count = frame_count
    self._sequence = 0
    self._image = np.full(self._shape, 255, dtype=np.uint8)
    for y, x, height, width in self._spot_declarations:
      self._image[y:y + height, x:x + width] = 0

  def prepare(self) -> None:
    """Returns configured spots and allocates the shared image buffer."""

    spots = SpotsBoxes()
    spots.set_spots(list(self._spot_declarations))
    spots.save_length()

    requesters = list()
    for request in self.config_requests_in:
      self.send_config(request, (spots, 128))
      requesters.append(request.requester)

    super().prepare()

    with self._artifact_path.open('w', encoding='utf-8') as file:
      json.dump({'requesters': requesters,
                 'memory_name': self._out_link_data.memory_name,
                 'shape': self._shape,
                 'dtype': 'uint8'}, file)

  def loop(self) -> None:
    """Publishes a static high-contrast image and eventually stops."""

    sequence = self._sequence
    self.send_img({'t(s)': time() - self.t0,
                   'ImageUniqueID': sequence,
                   'sequence': sequence},
                  self._image)
    self._sequence += 1

    if self._sequence >= self._frame_count:
      self.stop()


class BrokenConfiguredImageSource(VisionBlock):
  """Fails while a downstream Block waits for required configuration."""

  def __init__(self) -> None:
    """Sets a valid output format so only configuration handling can fail."""

    super().__init__(img_shape=(12, 16), img_dtype='uint8', debug=False)
    self.name = 'Broken configured image source'

  def prepare(self) -> None:
    """Raises before answering the downstream configuration request."""

    raise RuntimeError("Deliberate required configuration source failure")

  def loop(self) -> None:
    """Provides the concrete loop required by VisionBlock."""

    ...


def build_vision_camera_recorder_fanout(
    output_dir: Path) -> tuple[Block, ...]:
  """Builds CameraSource -> two ImageRecorders and saved notifications."""

  camera = CameraSource(
    '',
    image_generator=generate_vision_test_image,
    config=False,
    allow_downstream_config=False,
    img_shape=(48, 64),
    img_dtype='uint8',
    freq=30)

  fast_recorder = ImageRecorder(
    save_folder=output_dir / 'vision_fast_images',
    save_period=1,
    save_backend='npy',
    freq=80)

  sparse_recorder = ImageRecorder(
    save_folder=output_dir / 'vision_sparse_images',
    save_period=3,
    save_backend='npy',
    freq=80)

  notification_recorder = crappy.blocks.Recorder(
    output_dir / 'saved_notifications.csv',
    labels=('t(s)', 'img_index'),
    delay=0.05,
    freq=50)

  stop = crappy.blocks.StopBlock('t(s) > 0.8', freq=50)

  crappy.img_link(camera, fast_recorder, name='camera-fast-images')
  crappy.img_link(camera, sparse_recorder, name='camera-sparse-images')
  crappy.link(sparse_recorder, notification_recorder)
  crappy.link(camera, stop)

  return (camera, fast_recorder, sparse_recorder, notification_recorder,
          stop)


def build_vision_required_config_fanout(
    output_dir: Path) -> tuple[Block, ...]:
  """Builds a configured finite source feeding two independent probes."""

  source = FiniteConfiguredImageSource(output_dir / 'config_source.json')
  first_probe = ConfiguredFrameProbe(output_dir / 'config_probe_a.json',
                                     token='token-a',
                                     name='Config probe A')
  second_probe = ConfiguredFrameProbe(output_dir / 'config_probe_b.json',
                                      token='token-b',
                                      name='Config probe B')

  crappy.img_link(source, first_probe, name='configured-image-a')
  crappy.img_link(source, second_probe, name='configured-image-b')

  return source, first_probe, second_probe


def build_vision_dicve_recorder(output_dir: Path) -> tuple[Block, ...]:
  """Builds Generator -> CameraSource -> DICVEProcessor -> Recorder."""

  generator = _make_strain_generator()
  camera = _make_strain_camera()
  dicve = DICVEProcessor(
    patches=((24, 24, 32, 32), (72, 72, 32, 32)),
    request_configuration=True,
    method='Parabola',
    safe=True,
    follow=False,
    raise_on_patch_exit=False,
    freq=30)
  recorder = crappy.blocks.Recorder(
    output_dir / 'vision_dicve.csv',
    labels=('t(s)', 'Eyy(%)', 'Exx(%)'),
    delay=0.1,
    freq=30)

  crappy.link(generator, camera)
  crappy.img_link(camera, dicve, name='camera-dicve-images')
  crappy.link(dicve, recorder)

  return generator, camera, dicve, recorder


def build_vision_dis_correl_recorder(
    output_dir: Path) -> tuple[Block, ...]:
  """Builds Generator -> CameraSource -> DISCorrelProcessor -> Recorder."""

  generator = _make_strain_generator()
  camera = _make_strain_camera()
  dis_correl = DISCorrelProcessor(
    patch=(24, 24, 80, 80),
    fields=('exx', 'eyy'),
    labels=('t(s)', 'meta', 'Exx(%)', 'Eyy(%)'),
    request_configuration=True,
    finest_scale=1,
    iterations=1,
    gradient_iterations=5,
    residual=False,
    freq=30)
  recorder = crappy.blocks.Recorder(
    output_dir / 'vision_dis_correl.csv',
    labels=('t(s)', 'Exx(%)', 'Eyy(%)'),
    delay=0.1,
    freq=30)

  crappy.link(generator, camera)
  crappy.img_link(camera, dis_correl, name='camera-dis-correl-images')
  crappy.link(dis_correl, recorder)

  return generator, camera, dis_correl, recorder


def build_vision_video_extenso_recorder(
    output_dir: Path) -> tuple[Block, ...]:
  """Builds a configured image source -> VideoExtensoProcessor pipeline."""

  source = FiniteSpotImageSource(
    output_dir / 'video_extenso_source.json')
  video_extenso = VideoExtensoProcessor(
    white_spots=False,
    num_spots=2,
    min_area=100,
    blur=None,
    update_thresh=False,
    safe_mode=False,
    border=5,
    raise_on_lost_spot=True,
    freq=40)
  video_extenso.name = 'Video extenso processor'
  recorder = crappy.blocks.Recorder(
    output_dir / 'vision_video_extenso.csv',
    labels=('t(s)', 'Coord(px)', 'Eyy(%)', 'Exx(%)'),
    delay=0.05,
    freq=40)

  crappy.img_link(source, video_extenso,
                  name='camera-video-extenso-images')
  crappy.link(video_extenso, recorder)

  return source, video_extenso, recorder


def build_vision_broken_required_config(
    output_dir: Path) -> tuple[Block, ...]:
  """Builds a source failure while a probe awaits required configuration."""

  source = BrokenConfiguredImageSource()
  probe = ConfiguredFrameProbe(output_dir / 'unreachable_probe.json',
                               token='unreachable-token',
                               name='Waiting config probe')

  crappy.img_link(source, probe, name='broken-config-image')

  return source, probe
