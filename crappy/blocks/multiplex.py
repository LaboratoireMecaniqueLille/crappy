# coding: utf-8

from numpy import interp, array, searchsorted, concatenate
from itertools import chain

from .block import Block


class Multiplex(Block):
  """This block takes data from upstream blocks as input and interpolates it to
  output all labels in a common time basis.

  It is useful for synchronizing data acquired from different sensors, e.g. to
  plot a real-time stress-strain curve. This block is however quite
  resource-consuming, so it is preferable to perform interpolation in
  post-processing if real-time is not needed.

  Note:
    This block doesn't truly output data in real-time as it needs to wait for
    data from all the upstream blocks before performing the interpolation.
    So it should only be used with care when it is an input of a
    decision-making block. This is especially true when the upstream blocks
    have very different sample rates.
  """

  def __init__(self,
               time_label: str = 't(s)',
               freq: float = 200,
               verbose: bool = False) -> None:
    """Sets the args and initializes the parent class.

    Args:
      time_label: The label carrying the time information.
      freq : The sample rate for the interpolation, and the target looping
        frequency for the block. If this value is set too high and your machine
        cannot keep up, the block will most likely lag.
      verbose: If :obj:`True`, prints information about the looping frequency
        of the block.
    """

    Block.__init__(self)

    # Initializing the attributes
    self._time_label = time_label
    self.freq = freq
    self.verbose = verbose
    self._t = 0
    self._dt = 1 / self.freq

    # Creating the different dicts holding information
    self._values = dict()
    self._timestamps = dict()
    self._labels_to_get = dict()
    self._label_to_link = dict()

  def begin(self) -> None:
    """Receiving the first data from the upstream blocks, and checking that it
    is valid.

    If part of the data is not valid, warning the user. For the valid data,
    initializing the different dicts with it.
    """

    for link in self.inputs:
      # First, receiving data from each incoming link
      data = link.recv_chunk()

      # Handling the case when the time label is absent from the data
      if self._time_label not in data:
        print(f'WARNING : Cannot multiplex data coming from link {link.name} '
              f'as it does not contain the time label ({self._time_label})')
        continue

      # Extracting the timestamps from the received data
      timestamp = data.pop(self._time_label)

      # Determining which labels to use for multiplexing and warning the user
      # if two similar labels found
      self._labels_to_get[link] = [label for label in data if label not in
                                   chain(*self._labels_to_get.values())]
      if len(self._labels_to_get[link]) != len(data):
        print(f"WARNING : Got identical label(s)" 
              f"""{tuple(label for label in data if label 
                         not in self._labels_to_get[link])}""" 
              f"from at least two different links, on of them won't be used "
              f"for multiplexing.")

      self._label_to_link.update({label: link for label in
                                  self._labels_to_get[link]})

      # Storing the timestamps for the link
      self._timestamps[link] = array(timestamp)
      # Storing the values for the labels that were kept
      self._values.update({label: array(values) for label, values in
                           data.items() if label in self._labels_to_get[link]})

  def loop(self) -> None:
    """Receives data, interpolates it, and sends it to the downstream
    blocks."""

    self._get_data()
    self._send_data()

  def finish(self) -> None:
    """Just sending any remaining data."""

    self._send_data()

  def _get_data(self) -> None:
    """Receives data from the upstream links."""

    for link in self.inputs:
      # Receiving data from each link, non-blocking to prevent accumulation
      data = link.recv_chunk(blocking=False)
      # Processing only the valid labels
      if data is not None and link in self._labels_to_get:
        # Saving the timestamps
        self._timestamps[link] = concatenate((self._timestamps[link],
                                              array(data[self._time_label])))
        # Saving the other values
        for label in self._labels_to_get[link]:
          self._values[label] = concatenate((self._values[label],
                                             array(data[label])))

  def _send_data(self) -> None:
    """Interpolates the previously received data, and sends the result to the
    downstream blocks."""

    # Making sure all the necessary data has been received for interpolating
    if all(timestamp[-1] > self._t for timestamp
           in self._timestamps.values()) and any(
      timestamp.size for timestamp in self._timestamps.values()):

      # Getting the maximum timestamp common to all the labels
      max_t = min(times[-1] for times in self._timestamps.values())
      # Deducing the number of time intervals to interpolate on
      n_samples = int((max_t - self._t) // self._dt) + 1
      # Creating the array of timestamps for interpolation
      new_times = [self._t + i * self._dt for i in range(n_samples)]

      # For each link, getting the index for trimming data after interpolation
      last_indexes = {link: searchsorted(self._timestamps[link], new_times[-1],
                                         side='right')
                      for link in self.inputs}

      # This dict stores the values to send
      to_send = {self._time_label: new_times}

      # For each label, interpolating the data
      for label in self._values:
        link = self._label_to_link[label]
        to_send[label] = list(interp(new_times, self._timestamps[link],
                                     self._values[label]))

        # Trimming the values to save some memory
        self._values[label] = self._values[label][last_indexes[link]:]

      # Also trimming the timestamps to save some memory
      for link in self.inputs:
        self._timestamps[link] = self._timestamps[link][last_indexes[link]:]

      # Updating the current time value
      self._t = new_times[-1] + self._dt

      # Finally, sending the data to downstream blocks
      while any(to_send.values()):
        self.send({label: values.pop(0) for label, values in to_send.items()})
