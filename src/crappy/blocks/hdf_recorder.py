# coding: utf-8

import numpy as np
from typing import Union, Optional
from pathlib import Path
import logging

from .._global import OptionalModule
from .meta_block import Block

try:
  import tables
except ModuleNotFoundError:
  tables = OptionalModule("tables", "HDFRecorder needs the tables module to "
                          "write hdf files.")


class HDFRecorder(Block):
  """This Block records data efficiently into a HDF5 file.
  
  It expects data as :obj:`numpy.array` from exactly one upstream Block, that
  should be an :class:`~crappy.blocks.IOBlock` in `streamer` mode. It then 
  saves this data in a HDF5 file using the :mod:`tables` module.
  
  This Block is intended for high-speed data recording from 
  :class:`~crappy.inout.InOut` in `streamer` mode. For regular data recording,
  the :class:`~crappy.blocks.Recorder` Block should be used instead.
  
  Warning:
    Corrupted HDF5 files are not readable at all ! If anything goes wrong 
    during a test, especially during the finish phase, it is not guaranteed 
    that the recorded data will be readable.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.0.0 renamed from *Hdf_recorder* to *HDFRecorder*
  """

  def __init__(self,
               filename: Union[str, Path],
               node: str = 'table',
               expected_rows: int = 10**8,
               atom=None,
               label: str = 'stream',
               metadata: Optional[dict] = None,
               freq: Optional[float] = None,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      filename: Path to the output file, either relative or absolute. If the
        parent folders of the file do not exist, they will be created. If the
        file already exists, the actual file where data will be written will be
        renamed with a trailing index to avoid overriding it.
      node: The name of the array to create in the HDF5 file, as a :obj:`str`.
      expected_rows: The number of expected rows in the file. It is used to
        optimize the dumping.
      atom: This represents the type of data to be stored in the table. It can
        be given as a :obj:`tables.Atom` instance, as a :obj:`numpy.array`
        or as a :obj:`str`.
      label: The label carrying the data to be saved
      metadata: A :obj:`dict` containing additional information to save in the
        `HDF5` file.
      freq: The target looping frequency for the Block. If :obj:`None`, loops 
        as fast as possible.
        
        .. versionadded:: 1.5.10
      display_freq: if :obj:`True`, displays the looping frequency of the 
        Block.
        
        .. versionadded:: 1.5.10
        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      debug: If :obj:`True`, displays all the log messages including the
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
        
        .. versionadded:: 2.0.0
    """

    self._hfile = None

    super().__init__()
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    self._path = Path(filename)
    self._label = label
    self._metadata = {} if metadata is None else metadata
    self._expected_rows = expected_rows

    self._node = node
    atom = tables.Int16Atom() if atom is None else atom
    if not isinstance(atom, tables.Atom):
      self._atom = tables.Atom.from_dtype(np.dtype(atom))
    else:
      self._atom = atom

    self._array_initialized = False

  def prepare(self) -> None:
    """Checks that the Block has the right number of inputs, creates the
    folder containing the file if it doesn't already exist, changes the name of
    the file if it already exists, and initializes the HDF file."""

    # Making sure there's the right number of incoming links
    if not self.inputs:
      raise ValueError('The HDF Recorder block does not have inputs !')
    elif len(self.inputs) > 1:
      raise ValueError('Cannot link more than one block to an HDF Recorder '
                       'block !')

    parent_folder = self._path.parent

    # Creating the folder for storing the data if it does not already exist
    if not Path.is_dir(parent_folder):
      self.log(logging.INFO, f"Creating the folder containing the file to save"
                             f" data to ({parent_folder})")
      Path.mkdir(parent_folder, exist_ok=True, parents=True)

    # Changing the name of the file if it already exists
    if Path.exists(self._path):
      self.log(logging.WARNING, f"The file {self._path} already exists !")
      stem, suffix = self._path.stem, self._path.suffix
      i = 1
      # Adding an integer at the end of the name to identify the file
      while Path.exists(parent_folder / f'{stem}_{i:05d}{suffix}'):
        i += 1
      self._path = parent_folder / f'{stem}_{i:05d}{suffix}'
      self.log(logging.WARNING, f"Writing data to the file {self._path} "
                                f"instead !")

    # Initializing the file to save data to
    self.log(logging.INFO, "Initializing the HDF5 file")
    self._hfile = tables.open_file(str(self._path), "w")
    for name, value in self._metadata.items():
      self._hfile.create_array(self._hfile.root, name, value)

  def loop(self) -> None:
    """Receives data from the upstream Block and saves it.

    Also creates the array for recording data when the first values are
    received.
    """

    # Do nothing until the first value to save are received
    if not self._array_initialized:
      if self.data_available():
        self._first_loop()
        self._array_initialized = True
      else:
        return

    if self._label in (data := self.recv_all_data()):
      for elt in data[self._label]:
        self._array.append(elt)

  def finish(self) -> None:
    """Closes the HDF file."""

    if self._hfile is not None:
      self.log(logging.INFO, "Closing the HDF5 file")
      self._hfile.close()

  def _first_loop(self) -> None:
    """Initializes the array for saving data."""

    if self._label not in (data := self.recv_all_data()):
      raise KeyError(f'The data received by the HDF Recorder block does not '
                     f'contain the label {self._label} !')

    self.log(logging.INFO, "Initializing the arrays in the HDF5 file")

    _, width, *_ = data[self._label][0].shape
    self._array = self._hfile.create_earray(self._hfile.root,
                                            self._node,
                                            self._atom,
                                            (0, width),
                                            expectedrows=self._expected_rows)
    for elt in data[self._label]:
      self._array.append(elt)
