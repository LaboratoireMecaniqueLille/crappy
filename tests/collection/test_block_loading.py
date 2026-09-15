# coding: utf-8

from unittest import TestCase
from unittest.mock import patch

import crappy.blocks.camera as camera_block_module
import crappy.blocks.ioblock as ioblock_module
import crappy.blocks.machine as machine_module
import crappy.blocks.vision.camera as vision_camera_module
from crappy._collection import CollectionEntry
from crappy.actuator.meta_actuator import Actuator
from crappy.blocks import Camera as CameraBlock, IOBlock, Machine
from crappy.blocks.vision import CameraSource
from crappy.camera.meta_camera import Camera
from crappy.inout.meta_inout import InOut


class TestCollectionBlockLoading(TestCase):
  """Regression tests for retaining lazy-load information in Blocks."""

  def test_machine_accepts_core_and_collection_actuators_together(self) -> None:
    """Regression: a missing collection class must not invalidate core ones."""

    name = 'CollectionActuatorForTest'
    entry = CollectionEntry(name, 'Actuator', 'tests.collection.actuator')
    cls = type(name, (Actuator,), {'__module__': entry.module})
    machine_module.actuator_dict.pop(name)
    self.addCleanup(machine_module.actuator_dict.pop, name, None)

    def registry_get(kind: str, requested: str):
      return entry if (kind, requested) == ('Actuator', name) else None

    def load(_, classes: dict[str, type]):
      classes[name] = cls
      return cls

    settings = [{'type': 'FakeDCMotor'}, {'type': name}]
    with patch.object(machine_module.collection_registry, 'get',
                      side_effect=registry_get):
      with patch.object(machine_module, 'load_collection_class',
                        side_effect=load):
        machine = Machine(settings)

    self.assertEqual(machine._collection_entries, [entry])

  def test_preloaded_collection_actuator_keeps_its_reload_entry(self) -> None:
    """Regression: a preloaded Actuator still needs its entry under spawn."""

    name = 'PreloadedCollectionActuatorForTest'
    entry = CollectionEntry(name, 'Actuator', 'tests.collection.actuator')
    type(name, (Actuator,), {'__module__': entry.module})
    self.addCleanup(machine_module.actuator_dict.pop, name, None)

    with patch.object(machine_module.collection_registry, 'get',
                      return_value=entry):
      machine = Machine([{'type': name}])

    self.assertEqual(machine._collection_entries, [entry])

  def test_preloaded_collection_inout_keeps_its_reload_entry(self) -> None:
    """Checks that a preloaded InOut retains its entry for spawn."""

    name = 'CollectionInOutForTest'
    entry = CollectionEntry(name, 'InOut', 'tests.collection.inout')
    type(name, (InOut,), {'__module__': entry.module})
    self.addCleanup(ioblock_module.inout_dict.pop, name, None)

    with patch.object(ioblock_module.collection_registry, 'get',
                      return_value=entry):
      block = IOBlock(name)

    self.assertIs(block._collection_entry, entry)

  def test_preloaded_collection_camera_keeps_its_reload_entry(self) -> None:
    """Checks that the legacy Camera Block retains its entry for spawn."""

    name = 'CollectionCameraForTest'
    entry = CollectionEntry(name, 'Camera', 'tests.collection.camera')
    type(name, (Camera,), {'__module__': entry.module})
    self.addCleanup(camera_block_module.camera_dict.pop, name, None)
    self.addCleanup(CameraBlock.cam_count.pop, name, None)

    with patch.object(camera_block_module.collection_registry, 'get',
                      return_value=entry):
      block = CameraBlock(name, config=False, img_shape=(1, 1),
                          img_dtype='uint8')

    self.assertIs(block._collection_entry, entry)

  def test_preloaded_collection_camera_source_keeps_reload_entry(self) -> None:
    """Checks that CameraSource retains its entry for spawn."""

    name = 'CollectionCameraSourceForTest'
    entry = CollectionEntry(name, 'Camera', 'tests.collection.camera_source')
    type(name, (Camera,), {'__module__': entry.module})
    self.addCleanup(vision_camera_module.camera_dict.pop, name, None)
    self.addCleanup(CameraSource.cam_count.pop, name, None)

    with patch.object(vision_camera_module.collection_registry, 'get',
                      return_value=entry):
      block = CameraSource(name, config=False, img_shape=(1, 1),
                           img_dtype='uint8')

    self.assertIs(block._collection_entry, entry)

  def test_legacy_camera_ignores_empty_name_with_image_generator(self) -> None:
    """Regression: a synthetic image makes the Camera name irrelevant."""

    block = CameraBlock('', image_generator=lambda *_: None, config=False,
                        img_shape=(1, 1), img_dtype='uint8')

    self.assertEqual(block._camera_name, 'Image Generator')

  def test_camera_source_ignores_empty_name_with_image_generator(self) -> None:
    """Regression: CameraSource documents an empty synthetic Camera name."""

    block = CameraSource('', image_generator=lambda *_: None, config=False,
                         img_shape=(1, 1), img_dtype='uint8')

    self.assertEqual(block._camera_name, 'Image Generator')
