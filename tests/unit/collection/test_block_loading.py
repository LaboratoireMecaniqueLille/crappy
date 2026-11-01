# coding: utf-8

from unittest import TestCase
from unittest.mock import patch

import crappy.blocks.camera as camera_block_module
import crappy.blocks.auto_drive_video_extenso as auto_drive_module
import crappy.blocks.ioblock as ioblock_module
import crappy.blocks.machine as machine_module
import crappy.blocks.vision.camera as vision_camera_module
from crappy._collection import CollectionEntry
from crappy.actuator.meta_actuator import Actuator
from crappy.blocks import AutoDriveVideoExtenso, Camera as CameraBlock, IOBlock, Machine
from crappy.blocks.vision import CameraSource
from crappy.camera.meta_camera import Camera
from crappy.inout.meta_inout import InOut


class TestCollectionBlockLoading(TestCase):
  """Collection selection retains classes, not worker reload information."""

  def test_machine_accepts_core_and_collection_actuators_together(self) -> None:
    """Regression: a missing collection class must not invalidate core ones."""

    name = 'CollectionActuatorForTest'
    entry = CollectionEntry(name, 'Actuator', 'tests.unit.collection.actuator')
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

    self.assertEqual(machine._actuator_classes,
                     (machine_module.actuator_dict['FakeDCMotor'], cls))

  def test_preloaded_collection_actuator_retains_its_class(self) -> None:
    """Both Actuator owners reuse an already registered collection class."""

    name = 'PreloadedCollectionActuatorForTest'
    entry = CollectionEntry(name, 'Actuator', 'tests.unit.collection.actuator')
    cls = type(name, (Actuator,), {'__module__': entry.module})
    self.addCleanup(machine_module.actuator_dict.pop, name, None)

    with patch.object(machine_module.collection_registry, 'get',
                      return_value=entry):
      with (patch.object(machine_module, 'load_collection_class') as load_machine,
            patch.object(auto_drive_module, 'load_collection_class') as load_auto):
        machine = Machine([{'type': name}])
        auto_drive = AutoDriveVideoExtenso({'type': name})
        load_machine.assert_not_called()
        load_auto.assert_not_called()

    self.assertEqual(machine._actuator_classes, (cls,))
    self.assertIs(auto_drive._actuator_class, cls)

  def test_preloaded_collection_inout_retains_its_class(self) -> None:
    """A preloaded InOut is retained without another collection load."""

    name = 'CollectionInOutForTest'
    entry = CollectionEntry(name, 'InOut', 'tests.unit.collection.inout')
    cls = type(name, (InOut,), {'__module__': entry.module})
    self.addCleanup(ioblock_module.inout_dict.pop, name, None)

    with (patch.object(ioblock_module.collection_registry, 'get', return_value=entry),
          patch.object(ioblock_module, 'load_collection_class') as load):
      block = IOBlock(name)
      load.assert_not_called()

    self.assertIs(block._inout_class, cls)

  def test_preloaded_collection_camera_retains_its_class(self) -> None:
    """The legacy Camera retains a preloaded collection class."""

    name = 'CollectionCameraForTest'
    entry = CollectionEntry(name, 'Camera', 'tests.unit.collection.camera')
    cls = type(name, (Camera,), {'__module__': entry.module})
    self.addCleanup(camera_block_module.camera_dict.pop, name, None)
    self.addCleanup(CameraBlock.cam_count.pop, name, None)

    with (patch.object(camera_block_module.collection_registry, 'get', return_value=entry),
          patch.object(camera_block_module, 'load_collection_class') as load):
      block = CameraBlock(name, config=False, img_shape=(1, 1),
                          img_dtype='uint8')
      load.assert_not_called()

    self.assertIs(block._camera_class, cls)

  def test_preloaded_collection_camera_source_retains_its_class(self) -> None:
    """CameraSource retains a preloaded collection class."""

    name = 'CollectionCameraSourceForTest'
    entry = CollectionEntry(name, 'Camera', 'tests.unit.collection.camera_source')
    cls = type(name, (Camera,), {'__module__': entry.module})
    self.addCleanup(vision_camera_module.camera_dict.pop, name, None)
    self.addCleanup(CameraSource.cam_count.pop, name, None)

    with (patch.object(vision_camera_module.collection_registry, 'get', return_value=entry),
          patch.object(vision_camera_module, 'load_collection_class') as load):
      block = CameraSource(name, config=False, img_shape=(1, 1),
                           img_dtype='uint8')
      load.assert_not_called()

    self.assertIs(block._camera_class, cls)

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
