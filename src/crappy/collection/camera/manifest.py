# coding: utf-8

from ..._collection import CollectionEntry

DRIVERS: tuple[CollectionEntry, ...] = (
  CollectionEntry(
      name="CameraGPhoto2",
      kind="Camera",
      module="crappy.collection.camera.gphoto2_camera"),
  CollectionEntry(
      name="RaspberryPiCamera",
      kind="Camera",
      module="crappy.collection.camera.raspberry_pi_camera"),
  CollectionEntry(
      name="SeekThermalPro",
      kind="Camera",
      module="crappy.collection.camera.seek_thermal_pro")
)
