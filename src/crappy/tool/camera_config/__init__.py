# coding: utf-8

""":class:`~crappy.camera.meta_camera.camera.Camera` configuration models, GUI
backends, and selection tools.

The base package contains abstract configuration classes shared by the
'tkinter' and 'pyqt' packages. The possible configuration windows are selected
by :class:`~crappy.blocks.vision.CameraSource` and the all-in-one
:class:`Camera Blocks <crappy.blocks.Camera>` during preparation. config_tools
contains the boxes, spot detector, zoom model, and histogram worker they use.
"""

from .base import CameraConfig
from .config_tools import Box, Overlay, SpotsBoxes, SpotsDetector, Zoom
from .tkinter import (TkinterCameraConfig, TkinterCameraConfigBoxes,
                      TkinterDICVEConfig, TkinterDISCorrelConfig,
                      TkinterVideoExtensoConfig)
from .factory import create_configurator
