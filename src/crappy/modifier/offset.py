# coding: utf-8

from typing import Union
from collections.abc import Iterable
import logging

from .meta_modifier import Modifier, T


class Offset(Modifier):
  """This Modifier offsets every value of the given labels by a constant. This
  constant is calculated so that for each label the first returned value is
  equal to a user-defined target.

  For example if for a given label the target is `6` and the first received
  value is `3`, the Modifier will add `3` to each value received over this
  label.

  This Modifier can be used for example when measuring a variable that should
  start at `0` but doesn't because of a sensor offset. It can also just be used
  to plot nicer figures. It is not very accurate as it is only based on a
  single data point for the offset calculation. The ``make_zero`` argument of
  the :class:`~crappy.blocks.IOBlock` is a better alternative if precision is
  required when offsetting a sensor.
  
  .. versionadded:: 1.5.10
  """

  def __init__(self,
               labels: Union[str, Iterable[str]],
               offsets: Union[float, Iterable[float]]) -> None:
    """Sets the args and initializes the parent class.

    Args:
      labels: The labels to offset. Can be given as a single label, a
        :obj:`list` of labels or a :obj:`tuple` of labels.
      offsets: For each label, the target for the first received value. Can be
        given as a single value, a :obj:`list` of values or a :obj:`tuple` of
        values.
    """

    super().__init__()

    # Handling the case when only one label needs to be offset
    if isinstance(labels, str):
      labels = (labels,)
    labels = tuple(labels)
    try:
      iter(offsets)
    except TypeError:
      offsets = (offsets,)
    offsets = tuple(offsets)

    # Checking that the number of offsets match the number of labels
    if len(offsets) != len(labels):
      raise ValueError("As many offsets as there are labels should be given.")

    # Associating each offset to its label
    self._offsets = {label: offset for label, offset in zip(labels, offsets)}

    self._compensations = None
    self._compensated = False

  def __call__(self, data: dict[str, T]) -> dict[str, T]:
    """If the compensations are not set, sets them, and then offsets the
    required labels.
    
    .. versionchanged:: 2.0.0 renamed from *evaluate* to *__call__*
    """

    self.log(logging.DEBUG, f"Received {data}")

    # During the first loop, calculating the compensation values
    if not self._compensated:
      self._compensations = {label: -data[label] + offset for label, offset in
                             self._offsets.items()}
      self._compensated = True

    # Compensating the data to match the target offset value
    for label in self._offsets:
      data[label] += self._compensations[label]

    self.log(logging.DEBUG, f"Sending {data}")
    return data
