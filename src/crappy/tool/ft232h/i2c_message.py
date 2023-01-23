# coding: utf-8

from __future__ import annotations
from typing import Optional, List


class I2CMessage:
  """Class that mimics the :obj:`i2c_msg` class of the :mod:`smbus2` module."""

  def __init__(self,
               type_: str,
               address: int,
               length: Optional[int] = None,
               buf: Optional[list] = None) -> None:
    """Simply sets the attributes of the class, that characterise the i2c
    message.

    Args:
      type_ (:obj:`str`): Either a read (`'r'`) or a write (`'w'`) message
      address (:obj:`int`): The address of the I2C slave.
      length (:obj:`int`, optional): For a read message, the number of bytes to
        read.
      buf (:obj:`list`, optional): For a write message, the list of bytes to be
        written.
    """

    if type_ not in ['r', 'w']:
      raise ValueError("type_ should be either 'r' or 'w' !")

    self.type = type_
    self.addr = address
    self.len = length if length else 0
    self.buf = buf if buf else []

  @classmethod
  def read(cls, address: int, length: int) -> I2CMessage:
    """Instantiates an :class:`i2c_msg_ft232h` object for reading bytes.

    Args:
      address (:obj:`int`): The address of the I2C slave.
      length (:obj:`int`): The number of bytes to read.
    """

    return cls(type_='r', address=address, length=length)

  @classmethod
  def write(cls, address: int, buf: list) -> I2CMessage:
    """Instantiates an :class:`i2c_msg_ft232h` object for writing bytes.

    Args:
      address (:obj:`int`): The address of the I2C slave.
      buf (:obj:`list`): The list of bytes to be written.
    """

    return cls(type_='w', address=address, buf=buf)

  @property
  def addr(self) -> int:
    """The address of the I2C slave."""

    return self._addr

  @addr.setter
  def addr(self, addr_: int) -> None:
    if not isinstance(addr_, int) or not 0 <= addr_ <= 127:
      raise ValueError("addr should be an integer between 0 and 127 !")
    self._addr = addr_

  @property
  def buf(self) -> List[int]:
    """The list of bytes to be written, or the list of bytes read."""

    return self._buf

  @buf.setter
  def buf(self, buf_: List[int]) -> None:
    if self.type == 'w' and not buf_:
      raise ValueError("buf can't be empty for a write operation !")
    self._buf = buf_

  @property
  def len(self) -> int:
    """The number of bytes to read."""

    return self._len

  @len.setter
  def len(self, len_: int) -> None:
    if self.type == 'r' and (not isinstance(len_, int) or not len_ > 0):
      raise ValueError("len cannot be zero for a read operation !")
    self._len = len_

  def __iter__(self) -> object:
    self._n = 0
    return self

  def __next__(self) -> int:
    try:
      item = self._buf[self._n]
    except IndexError:
      raise StopIteration
    self._n += 1
    return item
