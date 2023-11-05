# coding: utf-8

from .meta_block import Block


class AutoDrive(Block):
  """Empty class for signaling an object of version 1.5 whose name changed in
  version 2.0 and is now deprecated.

  The new name of the correct object to use is given.
  """

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Block was renamed "
                              f"to AutoDriveVideoExtenso in version 2.0.0 ! "
                              f"Check the documentation for more information.")


class Client_server(Block):
  """Empty class for signaling an object of version 1.5 whose name changed in
  version 2.0 and is now deprecated.

  The new name of the correct object to use is given.
  """

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Block was renamed "
                              f"to ClientServer in version 2.0.0 ! "
                              f"Check the documentation for more information.")
  
  
class Displayer(Block):
  """Empty class for signaling an object of version 1.5 that wa removed in 
  version 2.0.O."""

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Block was removed "
                              f"in version 2.0.0 ! It is now contained "
                              f"directly in the Camera Block. "
                              f"Check the documentation for more information.")


class DISVE(Block):
  """Empty class for signaling an object of version 1.5 whose name changed in
  version 2.0 and is now deprecated.

  The new name of the correct object to use is given.
  """

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Block was renamed "
                              f"to DICVE in version 2.0.0 ! "
                              f"Check the documentation for more information.")


class Drawing(Block):
  """Empty class for signaling an object of version 1.5 whose name changed in
  version 2.0 and is now deprecated.

  The new name of the correct object to use is given.
  """

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Block was renamed "
                              f"to Canvas in version 2.0.0 ! "
                              f"Check the documentation for more information.")


class Fake_machine(Block):
  """Empty class for signaling an object of version 1.5 whose name changed in
  version 2.0 and is now deprecated.

  The new name of the correct object to use is given.
  """

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Block was renamed "
                              f"to FakeMachine in version 2.0.0 ! "
                              f"Check the documentation for more information.")


class GUI(Block):
  """Empty class for signaling an object of version 1.5 whose name changed in
  version 2.0 and is now deprecated.

  The new name of the correct object to use is given.
  """

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Block was renamed "
                              f"to Button in version 2.0.0 ! "
                              f"Check the documentation for more information.")
  
  
class Hdf_recorder(Block):
  """Empty class for signaling an object of version 1.5 whose name changed in
  version 2.0 and is now deprecated.

  The new name of the correct object to use is given.
  """

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Block was renamed "
                              f"to HDFRecorder in version 2.0.0 ! "
                              f"Check the documentation for more information.")


class Mean_block(Block):
  """Empty class for signaling an object of version 1.5 whose name changed in
  version 2.0 and is now deprecated.

  The new name of the correct object to use is given.
  """

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Block was renamed "
                              f"to MeanBlock in version 2.0.0 ! "
                              f"Check the documentation for more information.")


class Multiplex(Block):
  """Empty class for signaling an object of version 1.5 whose name changed in
  version 2.0 and is now deprecated.

  The new name of the correct object to use is given.
  """

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Block was renamed "
                              f"to Multiplexer in version 2.0.0 ! "
                              f"Check the documentation for more information.")


class Reader(Block):
  """Empty class for signaling an object of version 1.5 whose name changed in
  version 2.0 and is now deprecated.

  The new name of the correct object to use is given.
  """

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Block was renamed "
                              f"to LinkReader in version 2.0.0 ! "
                              f"Check the documentation for more information.")


class Video_extenso(Block):
  """Empty class for signaling an object of version 1.5 whose name changed in
  version 2.0 and is now deprecated.

  The new name of the correct object to use is given.
  """

  def __init__(self, *_, **__) -> None:
    """Simply raises the exception when instantiating the object."""

    super().__init__()

    raise NotImplementedError(f"The {type(self).__name__} Block was renamed "
                              f"to VideoExtenso in version 2.0.0 ! "
                              f"Check the documentation for more information.")
