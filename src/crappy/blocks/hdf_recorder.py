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
  """This block saves data efficiently into a hdf5 file.

  This block is meant to save data coming as arrays at a high rate
  (`>1kHz`). It relies on the module :mod:`tables`.

  Important:
    Do not forget to specify the type of data to be saved (see ``atom``
    parameter) to avoid casting the data into another type, as this could
    result in data loss or inefficient saving.
  """

  def __init__(self,
               filename: Union[str, Path],
               node: str = 'table',
               expected_rows: int = 10**8,
               atom=None,
               label: str = 'stream',
               metadata: Optional[dict] = None,
               freq: Optional[float] = None,
               verbose: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the args and initializes the parent class.

    Args:
      filename: Path to the output file, either relative or absolute. If the
        parent folders of the file do not exist, they will be created. If the
        file already exists, the actual file where data will be written will be
        renamed with a trailing index to avoid overriding it.
      expected_rows: The number of expected rows in the file. It is used to
        optimize the dumping.
      atom: This represents the type of data to be stored in the table. It can
        be given as a :class:`tables.Atom` instance, as a :class:`numpy.array`
        or as a :obj:`str`.
      label: The label carrying the data to be saved
      metadata: A :obj:`dict` containing additional information to save in the
        `hdf5` file.
      freq: The block will try to loop at this frequency.
      verbose: If :obj:`True`, displays the looping frequency of the block.
    """

    self._hfile = None

    super().__init__()
    self.freq = freq
    self.verbose = verbose
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
    """Checking that the block has the right number of inputs, creates the
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
    """Simply receives data from the upstream block and saves it.

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

    data = self.recv_all_data()

    if self._label in data:
      for elt in data[self._label]:
        self._array.append(elt)

  def finish(self) -> None:
    """Simply closes the HDF file."""

    if self._hfile is not None:
      self.log(logging.INFO, "Closing the HDF5 file")
      self._hfile.close()

  def _first_loop(self) -> None:
    """Initializes the array for saving data."""

    data = self.recv_all_data()

    if self._label not in data:
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
