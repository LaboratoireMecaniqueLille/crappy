# coding: utf-8

from .meta_modifier import Modifier

# Do not forget to revert changes in MetaModifier


class Moving_avg(Modifier):
  """Empty class for signaling an object of version 1.5 whose name changed in
    version 2.0 and is now deprecated.

  The new name of the correct object to use is given.
  """

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Modifier was "
                              f"renamed to MovingAvg in version 2.0.0 ! "
                              f"Check the documentation for more information.")


class Moving_med(Modifier):
  """Empty class for signaling an object of version 1.5 whose name changed in
    version 2.0 and is now deprecated.

  The new name of the correct object to use is given.
  """

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Modifier was "
                              f"renamed to MovingMed in version 2.0.0 ! "
                              f"Check the documentation for more information.")


class Trig_on_change(Modifier):
  """Empty class for signaling an object of version 1.5 whose name changed in
    version 2.0 and is now deprecated.

  The new name of the correct object to use is given.
  """

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Modifier was "
                              f"renamed to TrigOnChange in version 2.0.0 ! "
                              f"Check the documentation for more information.")


class Trig_on_value(Modifier):
  """Empty class for signaling an object of version 1.5 whose name changed in
    version 2.0 and is now deprecated.

  The new name of the correct object to use is given.
  """

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Modifier was "
                              f"renamed to TrigOnValue in version 2.0.0 ! "
                              f"Check the documentation for more information.")
