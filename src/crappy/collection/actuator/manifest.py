# coding: utf-8

from ..._collection import CollectionEntry

DRIVERS: tuple[CollectionEntry, ...] = (
  CollectionEntry(
      name="DCMotorHat",
      kind="Actuator",
      module="crappy.collection.actuator.adafruit_dc_motor_hat"),
  CollectionEntry(
      name="NewportTRA6PPD",
      kind="Actuator",
      module="crappy.collection.actuator.newport_tra6ppd"),
  CollectionEntry(
      name="OrientalARDK",
      kind="Actuator",
      module="crappy.collection.actuator.oriental_ard_k"),
  CollectionEntry(
      name="SchneiderMDrive23",
      kind="Actuator",
      module="crappy.collection.actuator.schneider_mdrive_23"),
)
