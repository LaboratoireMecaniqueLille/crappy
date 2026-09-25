# coding: utf-8

from collections.abc import Sequence
from collections import defaultdict
from pathlib import Path
import csv
import logging
from numbers import Real
from math import isfinite
from typing import Any
from time import monotonic

from .meta_block import Block


class Recorder(Block):
  """This Block saves data from an upstream Block to a CSV file.

  Incoming values are continuously buffered and periodically written as rows.
  The first row contains the names of the saved labels. Values are handled by
  :mod:`csv`, so in particular :obj:`None` is written as an empty field.

  This Block can only save data coming from exactly one upstream Block. To save
  data from multiple Blocks, use several instances of Recorder (recommended)
  or a :class:`~crappy.blocks.Multiplexer` Block.
  
  This Block cannot directly record data from "streams", i.e. coming from an
  :class:`~crappy.blocks.IOBlock` Block with the ``'streamer'`` argument set to
  :obj:`True`. To do so, the :class:`~crappy.blocks.HDFRecorder` Block should
  be used instead. Alternatively, a :class:`~crappy.modifier.Demux` Modifier 
  can be placed between the IOBlock and the Recorder, but most of the acquired
  data won't be saved.
  
  .. versionadded:: 1.4.0
  .. versionchanged:: 2.1.0 receive data without blocking, buffer it between
     loop iterations, and flush the remaining complete rows at shutdown
  """

  def __init__(self,
               file_name: str | Path,
               delay: float = 2,
               labels: str | Sequence[str] | None = None,
               freq: float | None = 200,
               display_freq: bool = False,
               debug: bool | None = False) -> None:
    """Sets the arguments and initializes the parent class.

    Args:
      file_name: Path to the output file, either relative or absolute. If the
        parent folders of the file do not exist, they will be created. If the
        file already exists, the actual output file is renamed with a trailing
        index to avoid overwriting it. Existing directories are rejected. The
        selected path is reserved during :meth:`prepare`.

        .. versionchanged:: 2.0.0 renamed from *filename* to *file_name*
      delay: Minimum delay between periodic writes, in seconds, as a finite,
        strictly positive number. Data received between writes is buffered. If
        no new data arrives after the delay expires, the pending data remains
        buffered until new data arrives or the Block finishes.
      labels: A non-empty label or sequence of non-empty labels to save. If
        :obj:`None`, the labels present in the first write window become the
        definitive set of recorded labels, new labels received later are
        ignored with a warning.
      freq: The target looping frequency for the Block. This determines how
        often the incoming Link is drained into the write buffer. If
        :obj:`None`, loops as fast as possible.
        
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

    match file_name:
      case Path() if file_name.name:
        self._path: Path = file_name
      case Path():
        raise ValueError("file_name must contain a file name, not a directory")
      case str() if file_name.strip() and Path(file_name).name:
        self._path: Path = Path(file_name)
      case str():
        raise ValueError("file_name must be non-empty when provided as a str, "
                         "and correspond to a file not to a directory")
      case _:
        raise TypeError("file_name must be provided as a non-empty string or "
                        "a Path")
    match delay:
      case Real() if not isfinite(delay) or delay <= 0:
        raise ValueError("delay must be a finite strictly positive float")
      case Real():
        self._delay: float = delay
      case _:
        raise TypeError("delay must be a finite strictly positive float")
    match labels:
      case None:
        self._requested_labels: list[str] | None = labels
      case str() if labels.strip():
        self._requested_labels: list[str] | None = [labels]
      case str():
        raise ValueError("labels must be a non-empty string or a sequence "
                         "of non-empty strings or None")
      case ():
        raise ValueError("labels must contain at least one element when "
                         "provided as a sequence")
      case (*labels,) if (all(isinstance(label, str) for label in labels) and
                          all(label.strip() for label in labels)):
        self._requested_labels: list[str] | None = list(labels)
      case (*_,):
        raise ValueError("labels must be a non-empty string or a sequence "
                         "of non-empty strings or None")
      case _:
        raise TypeError("labels must be a non-empty string or a sequence "
                        "of non-empty strings or None")

    self._recorder_labels: list[str] = (
      list() if self._requested_labels is None else self._requested_labels)
    self._file_initialized: bool = False
    self._data_buf: dict[str, list[Any]] = defaultdict(list)
    self._last_write_t: float = monotonic()

  def prepare(self) -> None:
    """Validates the Links and reserves an available output path.

    The parent folder is created when necessary. If the requested file already
    exists, a trailing index is added without overwriting any existing file.
    """

    # Making sure there's the right number of incoming links
    if not self.inputs:
      raise IOError('The Recorder block does not have inputs!')
    elif len(self.inputs) > 1:
      raise IOError('Cannot link more than one Block to a Recorder Block!')
    if self.outputs:
      raise IOError("The Recorder Block does not accept output Links!")

    # Check that the path doesn't point to a directory
    if self._path.is_dir():
      raise IsADirectoryError("The provided path points to an existing "
                              "directory!")

    # Creating the folder for storing the data if it does not already exist
    parent_folder = self._path.parent
    if not parent_folder.is_dir():
      self.log(logging.INFO, f"Creating the folder containing the file to save"
                             f" data to ({parent_folder})")
      parent_folder.mkdir(exist_ok=True, parents=True)

    # Change the name of the file with a suffix if it already exists, and
    # create the file without populating it yet
    try:
      self._path.touch(exist_ok=False)
    except FileExistsError:
      self.log(logging.WARNING, f"The file {self._path} already exists")
      stem, suffix = self._path.stem, self._path.suffix
      i = 1
      # Adding an integer at the end of the name to identify the file
      while True:
        try:
          (parent_folder / f'{stem}_{i:05d}{suffix}').touch(exist_ok=False)
          self._path = parent_folder / f'{stem}_{i:05d}{suffix}'
          self.log(logging.WARNING, f"Writing data to the file {self._path} "
                                    f"instead")
          break
        except FileExistsError:
          i += 1

  def begin(self) -> None:
    """Initializes the timer controlling periodic writes.

    .. versionadded:: 2.1.0
    """

    self._last_write_t = monotonic()

  def loop(self) -> None:
    """Buffers available input and writes it when the delay has elapsed.

    Only complete, rectangular rows are written. If no new input is available,
    this method leaves any buffered values for a later call or :meth:`finish`.
    """

    # Receiving data from each incoming link
    data = self.recv_all_data()

    if not data:
      self.log(logging.DEBUG, "No data received at this loop, returning")
      return

    # Store the received data in the write buffer
    for label, values in data.items():
      if self._requested_labels is None or label in self._recorder_labels:
        self._data_buf[label].extend(values)

    # If the write window is not complete yet, just return
    if monotonic() - self._last_write_t < self._delay:
      self.log(logging.DEBUG, "Not yet time to write data, returning")
      return

    # Write data if there is any to write
    self._write_file()

    # Clear the buffer and reset the time counter
    self._data_buf.clear()
    self._last_write_t = monotonic()

  def finish(self) -> None:
    """Flushes complete rows remaining in the write buffer."""

    self._write_file()

  def _write_file(self) -> None:
    """Validates the buffered columns and appends their rows to the CSV file.

    Every configured label must be present and contain the same number of
    values. The header is written before the first batch of rows.
    """

    # Passes if self._recorder_labels isn't configured yet
    if (self._data_buf and
        not all(label in self._data_buf for label in self._recorder_labels)):
      raise IOError("Not all labels to write received from upstream Block")

    # Passes if self._recorder_labels isn't configured yet
    if (self._data_buf and
        self._recorder_labels and
        self._requested_labels is None and
        not all(label in self._recorder_labels for label in self._data_buf)):
      extra = [label for label in self._data_buf
               if label not in self._recorder_labels]
      self.log(logging.WARNING, f"labels is None and the labels "
                                f"{', '.join(extra)} were received, but "
                                f"they're ignored since they were not present "
                                f"in the first received window of data")

    if self._data_buf:
      with open(self._path, 'a', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')

        # Write the headers first on the very first write
        if not self._file_initialized:
          # Get the definitive set of labels to track on the first write
          if self._requested_labels is None:
            self._recorder_labels = list(self._data_buf.keys())
          self.log(logging.INFO, f"Writing the header on file {self._path}")
          writer.writerow(self._recorder_labels)
          self._file_initialized = True

        # Sorting the lists of values in the same order as the labels
        sorted_data = [self._data_buf[label]
                       for label in self._recorder_labels]
        # Check that the number of points to write is consistent across labels
        if len({len(self._data_buf[label])
                for label in self._recorder_labels}) != 1:
          raise IOError("Data from different labels to write have different "
                        "numbers of points, cannot write to file")
        # Actually writing the values
        self.log(logging.DEBUG, f"Writing {len(sorted_data[0])} lines to the "
                                f"file {self._path}")
        writer.writerows(zip(*sorted_data, strict=True))
