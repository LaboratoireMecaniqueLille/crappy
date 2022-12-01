# coding: utf-8

from time import sleep
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from .block import Block


class Recorder(Block):
  """Saves data from an upstream block to a text file, with values separated by
  a coma and lines by a newline character.

  The first row of the file contains the names of the saved labels.
  This block can only save data coming from one upstream block. To save data
  from multiple blocks, use several instances of Recorder (recommended) or a
  :ref:`Multiplex` block.
  """

  def __init__(self,
               filename: Union[str, Path],
               delay: float = 2,
               labels: Optional[List[str]] = None,
               freq: float = 200,
               verbose: bool = True) -> None:
    """Sets the args and initializes the parent class.

    Args:
      filename: Path to the output file, either relative or absolute. If the
        parent folders of the file do not exist, they will be created. If the
        file already exists, the actual file where data will be written will be
        renamed with a trailing index to avoid overriding it.
      delay: Delay between each write in seconds.
      labels: If provided, only the data carried by these labels will be saved.
        Otherwise, all the received data is saved.
      freq: The block will try to loop at this frequency.
      verbose: If :obj:`True`, prints the looping frequency of the block.
    """

    super().__init__()
    self.niceness = -5
    self.freq = freq
    self.verbose = verbose

    self._delay = delay
    self._path = Path(filename)
    self._labels = labels

  def prepare(self) -> None:
    """Checking that the block has the right number of inputs, creates the
    folder containing the file if it doesn't already exist, and changes the
    name of the file if it already exists."""

    # Making sure there's the right number of incoming links
    if not self.inputs:
      raise ValueError('The Recorder block does not have inputs !')
    elif len(self.inputs) > 1:
      raise ValueError('Cannot link more than one block to a Recorder block !')
    self._link = self.inputs[0]

    parent_folder = self._path.parent

    # Creating the folder for storing the data if it does not already exist
    if not Path.is_dir(parent_folder):
      Path.mkdir(parent_folder, exist_ok=True, parents=True)

    # Changing the name of the file if it already exists
    if Path.exists(self._path):
      print(f'[Recorder] Warning ! The file {self._path} already exists !')
      stem, suffix = self._path.stem, self._path.suffix
      i = 1
      # Adding an integer at the end of the name to identify the file
      while Path.exists(parent_folder / f'{stem}_{i:05d}{suffix}'):
        i += 1
      self._path = parent_folder / f'{stem}_{i:05d}{suffix}'
      print(f'[Recorder] Using {self._path} instead !')

  def begin(self) -> None:
    """Receives the first chunk of data, writes the labels names in the first
    row of the file and starts saving the actual data."""

    data = self._link.recv_delay(self._delay)

    # If no labels are given, save everything that's received
    if self._labels is None:
      self._labels = list(data.keys())

    # The first row of the file contains the names of the labels
    with open(self._path, 'w') as file:
      file.write(f"{','.join(self._labels)}\n")

    # The following rows contain data
    self._save(data)

  def loop(self) -> None:
    """Simply receives data from the upstream block and saves it."""

    self._save(self._link.recv_delay(self._delay))

  def finish(self) -> None:
    """Gathers any data left in the links, and saves it."""

    sleep(0.5)
    data = self._link.recv_chunk_no_stop()
    if data is not None:
      self._save(data)

  def _save(self, data: Dict[str, List[Any]]):
    """Saves only the data carried by the specified labels to the file."""

    # Keeping only the data that needs to be saved
    data = {key: val for key, val in data.items() if key in self._labels}

    with open(self._path, 'a') as file:
      # Sorting the lists of values in the same order as the labels
      sorted_data = [data[label] for label in self._labels]
      # Actually writing the values
      for values in zip(*sorted_data):
        file.write(f"{','.join(map(str, values))}\n")


class Saver(Recorder):
  def __init__(self, *args, **kwargs) -> None:
    print('#### WARNING ####\n'
          'The block "Saver" has been renamed to "Recorder".\n'
          'Please replace the name in your program, '
          'it will be removed in future versions\n'
          '#################')
    super().__init__(*args, **kwargs)
