# coding: utf-8

from __future__ import annotations
from typing import Optional, Literal


class I2CMessage:
  """Class that mimics the :class:`smbus2.i2c_msg` class.

  It is used for communication with the
  :class:`~crappy.tool.ft232h.FT232HServer`, only by the
  :class:`~crappy.inout.MPRLS` InOut.
  
  .. versionadded:: 1.5.10
  .. versionchanged:: 2.0.0 renamed from *i2c_msg_ft232h* to *I2CMessage*
  """

  def __init__(self,
               type_: Literal['r', 'w'],
               address: int,
               length: Optional[int] = None,
               buf: Optional[list] = None) -> None:
    """Simply sets the attributes of the class, that characterizes the I2C
    message.

    Args:
      type_: Either a read (`'r'`) or a write (`'w'`) message
      address: The address of the I2C slave, as an :obj:`int`.
      length: For a read message, the number of bytes to read as an :obj:`int`.
      buf: For a write message, the :obj:`list` of :obj:`bytes` to be written.
    """

    if type_ not in ['r', 'w']:
      raise ValueError("type_ should be either 'r' or 'w' !")

    self.type = type_
    self.addr = address
    self.len = length if length else 0
    self.buf = buf if buf else []

  @classmethod
  def read(cls, address: int, length: int) -> I2CMessage:
    """Instantiates an :class:`I2CMessage` object for reading bytes.

    Args:
      address: The address of the I2C slave, as an :obj:`int`.
      length: The number of bytes to read, as an :obj:`int`.
    """

    return cls(type_='r', address=address, length=length)

  @classmethod
  def write(cls, address: int, buf: list) -> I2CMessage:
    """Instantiates an :class:`I2CMessage` object for writing bytes.

    Args:
      address: The address of the I2C slave, as an :obj:`int`.
      buf: The :obj:`list` of :obj:`bytes` to be written.
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
  def buf(self) -> list[int]:
    """The list of bytes to be written, or the list of bytes read."""

    return self._buf

  @buf.setter
  def buf(self, buf_: list[int]) -> None:
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
