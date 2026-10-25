# coding: utf-8

"""Shared argument checks for Crappy's public API."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any
from math import isfinite, isnan
from numbers import Integral, Real


def boolean(value: Any, name: str, optional: bool = False) -> None:
  """Checks that a value is a boolean.

  Args:
    value: Value to check. Only :class:`bool` instances are accepted, integers
      and NumPy boolean scalars are rejected.
    name: Argument name used in error messages, as a :obj:`str`.
    optional: Whether to also accept :obj:`None`. Defaults to :obj:`False`.

  Raises:
    TypeError: The value is not a boolean or an allowed :obj:`None`.

  .. versionadded:: 2.2.0
  """

  match value:
    case None if optional:
      return
    case bool():
      return
    case _:
      raise TypeError(f"{name} must be a boolean" +
                      (" or None" if optional else ""))


def real(value: Any,
         name: str,
         optional: bool = False,
         minimum: Real | None = None,
         maximum: Real | None = None,
         strict: bool = False,
         finite: bool = True) -> None:
  """Checks a real number, its finiteness, and optional bounds.

  Args:
    value: A :class:`numbers.Real` instance, including integers and compatible
      NumPy scalars. Booleans are rejected.
    name: Argument name used in error messages, as a :obj:`str`.
    optional: Whether to also accept :obj:`None`. Defaults to :obj:`False`.
    minimum: Inclusive lower bound, or :obj:`None` for no lower bound.
    maximum: Inclusive upper bound, or :obj:`None` for no upper bound.
    strict: If :obj:`True`, the lower bound is exclusive. It has no effect on
      the upper bound or when ``minimum`` is :obj:`None`.
    finite: Whether to reject infinities. Defaults to :obj:`True`. NaN is
      always rejected, and bounds still apply when infinities are allowed.

  Raises:
    TypeError: The value is not a real number, is a boolean, or is :obj:`None`
      when ``optional`` is :obj:`False`.
    ValueError: The value is NaN, is infinite when ``finite`` is :obj:`True`,
      exceeds the floating-point range used by the finiteness checks, or
      violates a bound.

  .. versionadded:: 2.2.0
  """

  match value:
    case None if optional:
      return
    case bool():
      raise TypeError(f"{name} must be a real number")
    case Real():
      try:
        invalid = isnan(value) or (finite and not isfinite(value))
      except OverflowError as error:
        raise ValueError(f"{name} must fit in a "
                         f"floating-point number") from error
      if invalid:
        raise ValueError(f"{name} must be " +
                         ("finite" if finite else "a number other than NaN"))
      if minimum is not None and (value < minimum or
                                  (strict and value == minimum)):
        raise ValueError(f"{name} must be "
                         f"{'greater than' if strict else 'at least'} "
                         f"{minimum}")
      if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    case _:
      raise TypeError(f"{name} must be a real number" +
                      (" or None" if optional else ""))


def integer(value: Any,
            name: str,
            optional: bool = False,
            minimum: Integral | None = None,
            maximum: Integral | None = None) -> None:
  """Checks an integer and optional inclusive bounds.

  Args:
    value: A :class:`numbers.Integral` instance, including NumPy integer
      scalars. Booleans and floats, including whole-valued floats, are
      rejected.
    name: Argument name used in error messages, as a :obj:`str`.
    optional: Whether to also accept :obj:`None`. Defaults to :obj:`False`.
    minimum: Inclusive lower bound, or :obj:`None` for no lower bound.
    maximum: Inclusive upper bound, or :obj:`None` for no upper bound.

  Raises:
    TypeError: The value is not an integer, is a boolean, or is :obj:`None`
      when ``optional`` is :obj:`False`.
    ValueError: The value is outside the supplied bounds.

  .. versionadded:: 2.2.0
  """

  match value:
    case None if optional:
      return
    case bool():
      raise TypeError(f"{name} must be an integer")
    case Integral():
      if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
      if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    case _:
      raise TypeError(f"{name} must be an integer" +
                      (" or None" if optional else ""))


def string(value: Any,
           name: str,
           optional: bool = False,
           empty: bool = False) -> None:
  """Checks a string, rejecting blank strings by default.

  Accepted strings are left unchanged, including any surrounding whitespace.

  Args:
    value: String to check.
    name: Argument name used in error messages, as a :obj:`str`.
    optional: Whether to also accept :obj:`None`. Defaults to :obj:`False`.
    empty: Whether to allow both empty and whitespace-only strings. Defaults
      to :obj:`False`.

  Raises:
    TypeError: The value is not a string or an allowed :obj:`None`.
    ValueError: The string is empty or whitespace-only and ``empty`` is
      :obj:`False`.

  .. versionadded:: 2.2.0
  """

  match value:
    case None if optional:
      return
    case str():
      if not empty and not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    case _:
      raise TypeError(f"{name} must be a string" +
                      (" or None" if optional else ""))


def choice(value: str,
           name: str,
           choices: Sequence,
           optional: bool = False) -> None:
  """Checks that a non-blank string is among the allowed choices.

  Args:
    value: String to check. Membership is tested without stripping whitespace
      or changing case.
    name: Argument name used in error messages, as a :obj:`str`.
    choices: Container of permitted strings, supporting membership tests.
    optional: Whether to also accept :obj:`None`, independently of ``choices``.
      Defaults to :obj:`False`.

  Raises:
    TypeError: The value is not a string or an allowed :obj:`None`.
    ValueError: The string is blank or is absent from ``choices``.

  .. versionadded:: 2.2.0
  """

  string(value, name, optional=optional)
  if value is not None and value not in choices:
    raise ValueError(f"{name} must be one of {choices}")


def callable_(value: Any, name: str, optional: bool = False) -> None:
  """Checks that a value is callable, without calling it.

  The callable's signature and return type are not checked.

  Args:
    value: Function, method, or other :class:`collections.abc.Callable`
      instance to check.
    name: Argument name used in error messages, as a :obj:`str`.
    optional: Whether to also accept :obj:`None`. Defaults to :obj:`False`.

  Raises:
    TypeError: The value is not callable or an allowed :obj:`None`.

  .. versionadded:: 2.2.0
  """

  match value:
    case None if optional:
      return
    case Callable():
      return
    case _:
      raise TypeError(f"{name} must be callable" +
                      (" or None" if optional else ""))


def sequence(value: Any,
             name: str,
             optional: bool = False,
             length: int | None = None,
             minimum: int = 0) -> None:
  """Checks a sequence and its length without checking its elements.

  Only :class:`collections.abc.Sequence` instances are accepted, excluding
  :class:`str`, :class:`bytes`, and :class:`bytearray`. Lists, tuples, and
  ranges qualify, generators, sets, and NumPy arrays do not.

  Args:
    value: Sequence to check.
    name: Argument name used in error messages, as a :obj:`str`.
    optional: Whether to also accept :obj:`None`. Defaults to :obj:`False`.
    length: Required exact length, or :obj:`None` to skip this constraint.
    minimum: Minimum number of elements, inclusive. Defaults to ``0``. This
      constraint also applies when ``length`` is provided.

  Raises:
    TypeError: The value is not an accepted sequence or an allowed :obj:`None`.
    ValueError: The sequence violates the exact or minimum length constraint.

  .. versionadded:: 2.2.0
  """

  match value:
    case None if optional:
      return
    case str() | bytes() | bytearray():
      raise TypeError(f"{name} must be a sequence, not a string")
    case Sequence():
      if length is not None and len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} elements")
      if len(value) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} elements")
    case _:
      raise TypeError(f"{name} must be a sequence" +
                      (" or None" if optional else ""))


def labels(value: Any,
           name: str,
           optional: bool = False,
           single: bool = True,
           empty: bool = False,
           unique: bool = True) -> None:
  """Checks one label or a sequence of non-empty string labels.

  The input is not converted to a list. Every label must be non-empty, even
  when an empty sequence is allowed. Duplicate comparisons use the original
  strings, including their case and whitespace.

  Args:
    value: A single label string or a sequence accepted by :func:`sequence`.
    name: Argument name used in error messages, as a :obj:`str`.
    optional: Whether to also accept :obj:`None`. Defaults to :obj:`False`.
    single: Whether a single string is accepted in place of a sequence.
      Defaults to :obj:`True`.
    empty: Whether an empty sequence is accepted. Defaults to :obj:`False`.
      This does not permit blank label strings.
    unique: Whether to reject repeated labels in a sequence. Defaults to
      :obj:`True`.

  Raises:
    TypeError: The input has an unsupported type or contains a non-string
      label.
    ValueError: A label is blank, the sequence is empty when ``empty`` is
      :obj:`False`, or labels repeat when ``unique`` is :obj:`True`.

  .. versionadded:: 2.2.0
  """

  if value is None and optional:
    return
  if single and isinstance(value, str):
    string(value, name)
    return
  sequence(value, name, minimum=0 if empty else 1)
  for label in value:
    string(label, name)
  if unique and len(set(value)) != len(value):
    raise ValueError(f"{name} must not contain duplicate labels")


def mapping(value: Any, name: str, optional: bool = False) -> None:
  """Checks a mapping with non-empty string keys.

  Empty mappings are accepted. The mapped values are not checked.

  Args:
    value: A :class:`collections.abc.Mapping` instance to check.
    name: Argument name used in error messages, as a :obj:`str`.
    optional: Whether to also accept :obj:`None`. Defaults to :obj:`False`.

  Raises:
    TypeError: The value is not a mapping or an allowed :obj:`None`, or a key
      is not a string.
    ValueError: A key is empty or whitespace-only.

  .. versionadded:: 2.2.0
  """

  match value:
    case None if optional:
      return
    case Mapping():
      for key in value:
        string(key, f"{name} keys")
    case _:
      raise TypeError(f"{name} must be a mapping" +
                      (" or None" if optional else ""))


def log_level(value: Integral,
              name: str = 'log_level',
              optional: bool = False) -> None:
  """Checks a non-negative integer logging level.

  Custom numeric levels are accepted, including ``0``. The value need not be
  one of the predefined :mod:`logging` constants. Boolean values and level
  names such as ``'DEBUG'`` are rejected.

  Args:
    value: Logging level to check, as a :class:`numbers.Integral` instance.
    name: Argument name used in error messages. Defaults to ``'log_level'``.
    optional: Whether to also accept :obj:`None`. Defaults to :obj:`False`.

  Raises:
    TypeError: The value is not an integer, is a boolean, or is :obj:`None`
      when ``optional`` is :obj:`False`.
    ValueError: The logging level is negative.

  .. versionadded:: 2.2.0
  """

  integer(value, name, optional=optional, minimum=0)


def instance(value: Any, name: str, cls: type, optional: bool = False) -> None:
  """Checks a value's class, accepting subclasses through :func:`isinstance`.

  Args:
    value: Object to check.
    name: Argument name used in error messages, as a :obj:`str`.
    cls: Required class. Pass a single class, as its ``__name__`` is used in
      error messages.
    optional: Whether to accept :obj:`None` without checking its class.
      Defaults to :obj:`False`.

  Raises:
    TypeError: The value is not an instance of ``cls`` or an allowed
      :obj:`None`.

  .. versionadded:: 2.2.0
  """

  if value is None and optional:
    return
  if not isinstance(value, cls):
    raise TypeError(f"{name} must be an instance of {cls.__name__}")


def interface(value: Any,
              name: str,
              methods: Sequence[str],
              optional: bool = False) -> None:
  """Checks that an object exposes the callable attributes of a protocol.

  Attributes are retrieved with :func:`getattr`. Methods are not called, and
  their signatures and return types are not checked. An :exc:`AttributeError`
  during lookup counts as a missing attribute, other exceptions propagate
  unchanged.

  Args:
    value: Object expected to implement the protocol.
    name: Argument name used in error messages, as a :obj:`str`.
    methods: Reusable collection of attribute names, as strings, for example
      a tuple or list. An empty collection imposes no requirements.
    optional: Whether to accept :obj:`None` without inspecting its attributes.
      Defaults to :obj:`False`.

  Raises:
    TypeError: At least one requested attribute is missing or is not callable.

  .. versionadded:: 2.2.0
  """

  if value is None and optional:
    return
  if any(not callable(getattr(value, method, None)) for method in methods):
    raise TypeError(f"{name} must provide {', '.join(methods)}")


def dtype(value: Any, name: str = 'img_dtype', optional: bool = False) -> None:
  """Checks a string describing a fixed-size NumPy dtype.

  The string is parsed with :class:`numpy.dtype`. Valid dtype descriptions
  are accepted even if non-numeric, provided their item size is positive and
  they contain no Python objects. NumPy dtype instances and scalar classes
  are not accepted in place of a string.

  Args:
    value: Non-blank NumPy dtype string to check.
    name: Argument name used in error messages. Defaults to ``'img_dtype'``.
    optional: Whether to also accept :obj:`None`. Defaults to :obj:`False`.

  Raises:
    TypeError: The value is not a string or an allowed :obj:`None`.
    ValueError: The string is blank, cannot be parsed as a dtype, describes
      a dtype containing objects, or has a zero item size.

  .. versionadded:: 2.2.0
  """

  if value is None and optional:
    return
  string(value, name)
  import numpy as np
  try:
    dt = np.dtype(value)
  except (TypeError, ValueError) as error:
    raise ValueError(f"{name} must describe a valid NumPy dtype") from error
  if dt.hasobject or dt.itemsize == 0:
    raise ValueError(f"{name} must have a fixed size and contain no objects")


def image(value: Any,
          name: str = 'image',
          optional: bool = False,
          ndim: Sequence[int] = (2, 3),
          minimum: int = 1,
          numeric: bool = True) -> None:
  """Checks a NumPy array's dimensions, sizes, and optionally numeric dtype.

  This does not inspect individual values: NaN and infinity are accepted.
  It does not convert the array or impose memory-layout or channel-count
  requirements.

  Args:
    value: A :class:`numpy.ndarray` instance, including subclasses.
    name: Argument name used in error messages. Defaults to ``'image'``.
    optional: Whether to also accept :obj:`None`. Defaults to :obj:`False`.
    ndim: Container of permitted numbers of dimensions. Defaults to ``(2, 3)``.
    minimum: Minimum size of each axis, inclusive. Defaults to ``1``, rejecting
      empty axes. Use ``0`` to permit them.
    numeric: Whether to require a dtype that is a subtype of
      :class:`numpy.number` and is not complex. Defaults to :obj:`True`.
      Boolean and object arrays are also rejected. If :obj:`False`, any dtype
      is accepted.

  Raises:
    TypeError: The value is not an array or an allowed :obj:`None`, or its
      dtype fails the numeric constraint.
    ValueError: The number of dimensions is not in ``ndim`` or an axis is
      smaller than ``minimum``.

  .. versionadded:: 2.2.0
  """

  if value is None and optional:
    return
  import numpy as np
  instance(value, name, np.ndarray)
  if value.ndim not in ndim or any(size < minimum for size in value.shape):
    raise ValueError(f"{name} must be a non-empty array of dimension {ndim}")
  if numeric and (not np.issubdtype(value.dtype, np.number) or
                  np.iscomplexobj(value)):
    raise TypeError(f"{name} must contain real numeric values")
