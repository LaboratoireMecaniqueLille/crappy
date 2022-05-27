# coding: utf-8

import numpy as np
from typing import Union, Optional
from pathlib import Path

from .._global import OptionalModule

try:
  import tables
except ModuleNotFoundError:
  tables = OptionalModule("tables", "Hdf_recorder needs the tables module to "
                          "write hdf files.")

from .block import Block


class Hdf_recorder(Block):
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
               verbose: bool = False) -> None:
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
      verbose: If :obj:`True`, prints the looping frequency of the block.
    """

    Block.__init__(self)
    if freq is not None:
      self.freq = freq
    self.verbose = verbose

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
    self._link = self.inputs[0]

    parent_folder = self._path.parent

    # Creating the folder for storing the data if it does not already exist
    if not Path.is_dir(parent_folder):
      Path.mkdir(parent_folder, exist_ok=True, parents=True)

    # Changing the name of the file if it already exists
    if Path.exists(self._path):
      print(f'[HDF Recorder] Warning ! The file {self._path} already exists !')
      stem, suffix = self._path.stem, self._path.suffix
      i = 1
      # Adding an integer at the end of the name to identify the file
      while Path.exists(parent_folder / f'{stem}_{i:05d}{suffix}'):
        i += 1
      self._path = parent_folder / f'{stem}_{i:05d}{suffix}'
      print(f'[HDF Recorder] Using {self._path} instead !')

    # Initializing the file to save data to
    self._hfile = tables.open_file(str(self._path), "w")
    for name, value in self._metadata.items():
      self._hfile.create_array(self._hfile.root, name, value)

  def begin(self) -> None:
    """Receives the first chunk of data, makes sure that it contains the label
    to save, and initializes the HDF array with it."""

    data = self._link.recv_chunk(blocking=True)

    if self._label not in data:
      raise KeyError(f'The data received by the HDF Recorder block does not '
                     f'contain the label {self._label} !')

    _, width, *_ = data[self._label][0].shape
    self._array = self._hfile.create_earray(self._hfile.root,
                                            self._node,
                                            self._atom,
                                            (0, width),
                                            expectedrows=self._expected_rows)
    for elt in data[self._label]:
      self._array.append(elt)

  def loop(self) -> None:
    """Simply receives data from the upstream block and saves it."""

    data = self._link.recv_chunk(blocking=False)

    if data is not None:
      for elt in data[self._label]:
        self._array.append(elt)

  def finish(self) -> None:
    """Simply closes the HDF file."""

    self._hfile.close()


class Hdf_saver(Hdf_recorder):
  def __init__(self, *args, **kwargs) -> None:
    print('#### WARNING ####\n'
          'The block "Hdf_saver" has been renamed to "Hdf_recorder".\n'
          'Please replace the name in your program, '
          'it will be removed in future versions\n'
          '#################')
    Hdf_recorder.__init__(self, *args, **kwargs)
