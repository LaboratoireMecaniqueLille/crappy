# coding: utf-8

from typing import Optional, Union
from collections.abc import Iterable
from pathlib import Path
import logging

from .meta_block import Block


class Recorder(Block):
  """This Block saves data from an upstream Block to a text file, with values 
  separated by a coma and lines by a newline character.

  The first row of the file contains the names of the saved labels. This Block 
  can only save data coming from exactly one upstream Block. To save data
  from multiple Blocks, use several instances of Recorder (recommended) or a
  :class:`~crappy.blocks.Multiplexer` Block.
  
  This Block cannot directly record data from "streams", i.e. coming from an
  :class:`~crappy.blocks.IOBlock` Block with the ``'streamer'`` argument set to
  :obj:`True`. To do so, the :class:`~crappy.blocks.HDFRecorder` Block should
  be used instead. Alternatively, a :class:`~crappy.modifier.Demux` Modifier 
  can be placed between the IOBlock and the Recorder, but most of the acquired
  data won't be saved.
  
  .. versionadded:: 1.4.0
  """

  def __init__(self,
               file_name: Union[str, Path],
               delay: float = 2,
               labels: Optional[Union[str, Iterable[str]]] = None,
               freq: Optional[float] = 200,
               display_freq: bool = False,
               debug: Optional[bool] = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      file_name: Path to the output file, either relative or absolute. If the
        parent folders of the file do not exist, they will be created. If the
        file already exists, the actual file where data will be written will be
        renamed with a trailing index to avoid overriding it.

        .. versionchanged:: 2.0.0 renamed from *filename* to *file_name*
      delay: Delay between each write in seconds.
      labels: If provided, only the data carried by these labels will be saved.
        Otherwise, all the received data is saved.
      freq: The target looping frequency for the Block. If :obj:`None`, loops 
        as fast as possible.
        
        .. versionadded:: 1.5.10
      display_freq: If :obj:`True`, displays the looping frequency of the
        Block.
        
        .. versionadded:: 1.5.10
        .. versionchanged:: 2.0.0 renamed from *verbose* to *display_freq*
      debug: If :obj:`True`, displays all the log messages including the 
        :obj:`~logging.DEBUG` ones. If :obj:`False`, only displays the log
        messages with :obj:`~logging.INFO` level or higher. If :obj:`None`,
        disables logging for this Block.
        
        .. versionadded:: 2.0.0
    """

    super().__init__()
    self.niceness = -5
    self.freq = freq
    self.display_freq = display_freq
    self.debug = debug

    self._delay = delay
    self._path = Path(file_name)

    # Forcing the labels into a list
    if labels is not None and isinstance(labels, str):
      self._labels = [labels]
    elif labels is not None:
      self._labels = list(labels)
    else:
      self._labels = None

    self._file_initialized = False

  def prepare(self) -> None:
    """Checks that the Block has the right number of inputs, creates the
    folder containing the file if it doesn't already exist, and changes the
    name of the file if it already exists."""

    # Making sure there's the right number of incoming links
    if not self.inputs:
      raise ValueError('The Recorder block does not have inputs !')
    elif len(self.inputs) > 1:
      raise ValueError('Cannot link more than one block to a Recorder block !')

    # Creating the folder for storing the data if it does not already exist
    if not Path.is_dir(parent_folder := self._path.parent):
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

  def loop(self) -> None:
    """Receives data from the upstream Block and saves it."""

    if not self._file_initialized:
      if self.data_available():

        data = self.recv_all_data(delay=self._delay)

        # If no labels are given, save everything that's received
        if self._labels is None:
          self._labels = list(data.keys())

        # The first row of the file contains the names of the labels
        with open(self._path, 'w') as file:
          self.log(logging.INFO, f"Writing the header on file {self._path}")
          file.write(f"{','.join(self._labels)}\n")

        self._file_initialized = True
      else:
        return

    else:
      data = self.recv_all_data(delay=self._delay)

    # Keeping only the data that needs to be saved
    data = {key: val for key, val in data.items() if key in self._labels}

    if data:
      with open(self._path, 'a') as file:
        # Sorting the lists of values in the same order as the labels
        sorted_data = [data[label] for label in self._labels]
        # Actually writing the values
        self.log(logging.DEBUG, f"Writing {sorted_data} to the file "
                                f"{self._path}")
        for values in zip(*sorted_data):
          file.write(f"{','.join(map(str, values))}\n")
