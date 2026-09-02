# coding: utf-8

from collections.abc import Callable
from pathlib import Path
from crappy.blocks import Block

from .numeric import (build_generator_recorder,
                      build_ioblock_recorder,
                      build_machine_recorder,
                      build_multiplexer_recorder,
                      build_pid_fake_machine_recorder,
                      build_stream_hdf_recorder,
                      build_synchronizer_recorder)
from .lifecycle import (build_auto_drive_recorder,
                        build_generator_link_reader,
                        build_generator_sink,
                        build_pause_stop_probe)
from .vision import (build_camera_image_saver,
                     build_dicve_recorder,
                     build_dis_correl_recorder)


scenario_builders: dict[str, Callable[[Path], tuple[Block, ...]]] = {
  'auto_drive_recorder': build_auto_drive_recorder,
  'camera_image_saver': build_camera_image_saver,
  'dicve_recorder': build_dicve_recorder,
  'dis_correl_recorder': build_dis_correl_recorder,
  'generator_link_reader': build_generator_link_reader,
  'generator_recorder': build_generator_recorder,
  'generator_sink': build_generator_sink,
  'ioblock_recorder': build_ioblock_recorder,
  'machine_recorder': build_machine_recorder,
  'multiplexer_recorder': build_multiplexer_recorder,
  'pause_stop_probe': build_pause_stop_probe,
  'pid_fake_machine_recorder': build_pid_fake_machine_recorder,
  'stream_hdf_recorder': build_stream_hdf_recorder,
  'synchronizer_recorder': build_synchronizer_recorder,
}
