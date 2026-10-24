# coding: utf-8

"""Bundled image resources loaded when their public namespace is requested."""

import importlib.resources
from numpy import frombuffer, uint8

from ._global import OptionalModule


class resources:
  """This class defines aliases for quick access to the resources in the
  `tool/data/` folder.

  These aliases are then used in the examples provided on the GitHub
  repository, but could also be used in custom user scripts.

  .. versionadded:: 1.5.3
  """

  try:
    # Defining aliases to the images themselves
    from cv2 import imdecode
    speckle_path = importlib.resources.files('crappy').joinpath(
        'tool/data/speckle.png')
    speckle = imdecode(frombuffer(speckle_path.read_bytes(), uint8),
                       flags=0)

    ve_markers_path = importlib.resources.files('crappy').joinpath(
        'tool/data/ve_markers.tif')
    ve_markers = imdecode(frombuffer(ve_markers_path.read_bytes(), uint8),
                          flags=0)

    pad_path = importlib.resources.files('crappy').joinpath(
        'tool/data/pad.png')
    pad = imdecode(frombuffer(pad_path.read_bytes(), uint8),
                   flags=0)

  # In case the module opencv-python is missing
  except (ModuleNotFoundError, ImportError) as error:
    speckle = OptionalModule('cv2', cause=error, install_name='opencv-python')
    ve_markers = OptionalModule('cv2', cause=error,
                                install_name='opencv-python')
    pad = OptionalModule('cv2', cause=error, install_name='opencv-python')

  # Also getting the paths to the images
  ref_pad = (importlib.resources.files('crappy') / 'tool' / 'data' /
             'pad.png')
  ref_speckle = (importlib.resources.files('crappy') / 'tool' / 'data' /
                 'speckle.png')
  ref_ve = (importlib.resources.files('crappy') / 'tool' / 'data' /
            've_markers.tif')
  with (importlib.resources.as_file(ref_pad) as path_pad,
        importlib.resources.as_file(ref_speckle) as path_speckle,
        importlib.resources.as_file(ref_ve) as path_ve):
    paths = {'pad': path_pad, 'speckle': path_speckle, 've_markers': path_ve}
