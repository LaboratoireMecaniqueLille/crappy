# coding: utf-8

from dataclasses import dataclass
from typing import Optional, List, Tuple

from .box import Box


@dataclass
class SpotsBoxes:
  """This class stores up to four instances of :class:`Box`, defining the
  bounding boxes of the spots for video extensometry or the patches for DISVE.

  It can also instantiate the Box object by parsing a list of tuples containing
  enough information.
  """

  spot_1: Optional[Box] = None
  spot_2: Optional[Box] = None
  spot_3: Optional[Box] = None
  spot_4: Optional[Box] = None

  _index = -1

  def __getitem__(self, i: int) -> Optional[Box]:
    if i == 0:
      return self.spot_1
    elif i == 1:
      return self.spot_2
    elif i == 2:
      return self.spot_3
    elif i == 3:
      return self.spot_4
    else:
      raise IndexError

  def __setitem__(self, i: int, value: Optional[Box]) -> None:
    if i == 0:
      self.spot_1 = value
    elif i == 1:
      self.spot_2 = value
    elif i == 2:
      self.spot_3 = value
    elif i == 3:
      self.spot_4 = value
    else:
      raise IndexError

  def __iter__(self):
    self._index = -1
    return self

  def __next__(self) -> Box:
    self._index += 1
    try:
      return self[self._index]
    except IndexError:
      raise StopIteration

  def __len__(self) -> int:
    return len([spot for spot in self if spot is not None])

  def set_spots(self,
                spots: List[Tuple[int, int, int, int]]) -> None:
    """Parses a list of tuples and instantiates the corresponding Box
    objects."""

    for i, spot in enumerate(spots):
      self[i] = Box(x_start=spot[1], x_end=spot[1] + spot[3],
                    y_start=spot[0], y_end=spot[0] + spot[2])

  def empty(self) -> bool:
    """Returns :obj:`True` if all spots are :obj:`None`, else :obj:`False`."""

    return all(spot is None for spot in self)

  def reset(self) -> None:
    """Resets the boxes to :obj:`None`."""

    for i in range(4):
      self[i] = None
