# coding: utf-8

"""Hardware-free Cameras shared by the base and backend tests."""

from time import time
from contextlib import contextmanager
from tests.support.assertions import exception_leaves
from tests.support.images import gradient_image, spots_image
from typing import Any
import numpy as np

from crappy.camera.meta_camera import Camera


class DummyCamera(Camera):
  """Subclass of Camera that does nothing, but Camera cannot be instantiated
  directly."""

  frame: tuple[dict[str, Any] | float, np.ndarray] | None = None

  def get_image(self) -> tuple[dict[str, Any] | float, np.ndarray] | None:
    """Mandatory to implement in subclasses."""

    return self.frame


class FakeTestCameraSimple(Camera):
  """Fake :class:`~crappy.camera.Camera` used for tests, generating a
  grey-level gradient image.

  .. versionadded:: 2.0.8
  """

  def __init__(self, min_val: int = 0, max_val: int = 255) -> None:
    """Initializes the parent class.

    Args:
      min_val: Minimum value in the generated image.
      max_val: Maximum value in the generated image.
    """

    super().__init__()

    self._min = min_val
    self._max = max_val

  def get_image(self) -> tuple[float, np.ndarray]:
    """Generates a grey-level image containing a gradient from the specified
    minimum to the specified maximum."""

    return time(), gradient_image(self._min, self._max)


class FakeTestCameraSpots(Camera):
  """Fake :class:`~crappy.camera.Camera` used for test of the
  video-extensometry configurator, generating a white image with four round
  spots.

  .. versionadded:: 2.0.8
  """

  def get_image(self) -> tuple[float, np.ndarray]:
    """Generates a white image with four round black spots."""

    return time(), spots_image()


class FakeTestCameraParams(Camera):
  """Fake :class:`~crappy.camera.Camera` used for testing the parameter
  handling in the configuration interface.

  .. versionadded:: 2.0.8
  """

  def __init__(self) -> None:
    """Initializes the parent class and sets the attributes."""

    super().__init__()

    self._bool_getter_called: bool = False
    self._bool_setter_called: bool = False
    self._scale_int_getter_called: bool = False
    self._scale_int_setter_called: bool = False
    self._scale_float_getter_called: bool = False
    self._scale_float_setter_called: bool = False
    self._choice_getter_called: bool = False
    self._choice_setter_called: bool = False

    self._scale_int_bounds = (-100, 100, 2)
    self._scale_float_bounds = (-10.0, 10.0, 0.1)
    self._choices = ('choice_1', 'choice_2', 'choice_3')

  def open(self) -> None:
    """Instantiates the four camera parameters to test."""

    self.add_bool_setting('bool_setting',
                          self._bool_getter,
                          self._bool_setter,
                          True)

    self.add_scale_setting('scale_int_setting',
                           self._scale_int_bounds[0],
                           self._scale_int_bounds[1],
                           self._scale_int_getter,
                           self._scale_int_setter,
                           default=0,
                           step=self._scale_int_bounds[2])

    self.add_scale_setting('scale_float_setting',
                           self._scale_float_bounds[0],
                           self._scale_float_bounds[1],
                           self._scale_float_getter,
                           self._scale_float_setter,
                           default=0.,
                           step=self._scale_float_bounds[2])

    self.add_choice_setting('choice_setting',
                            self._choices,
                            self._choice_getter,
                            self._choice_setter,
                            self._choices[0])

    # Left out on purpose
    # self.set_all()

  def get_image(self) -> tuple[dict[str, Any] | float, np.ndarray] | None:
    """Added because this method must be defined by children classes."""

    return super().get_image()

  def _bool_setter(self, value: bool) -> None:
    """Setter for the boolean parameter."""

    self._bool_setter_called = True
    self.settings['bool_setting']._value_no_getter = value

  def _bool_getter(self) -> bool:
    """Getter for the boolean parameter."""

    self._bool_getter_called = True
    return self.settings['bool_setting']._value_no_getter

  def _scale_int_setter(self, value: int) -> None:
    """Setter for the integer parameter."""

    self._scale_int_setter_called = True
    self.settings['scale_int_setting']._value_no_getter = value

  def _scale_int_getter(self) -> int:
    """Getter for the integer parameter."""

    self._scale_int_getter_called = True
    return self.settings['scale_int_setting']._value_no_getter

  def _scale_float_setter(self, value: float) -> None:
    """Setter for the float parameter."""

    self._scale_float_setter_called = True
    self.settings['scale_float_setting']._value_no_getter = value

  def _scale_float_getter(self) -> float:
    """Getter for the float parameter."""

    self._scale_float_getter_called = True
    return self.settings['scale_float_setting']._value_no_getter

  def _choice_setter(self, value: str) -> None:
    """Setter for the string parameter."""

    self._choice_setter_called = True
    self.settings['choice_setting']._value_no_getter = value

  def _choice_getter(self) -> str:
    """Getter for the string parameter."""

    self._choice_getter_called = True
    return self.settings['choice_setting']._value_no_getter


@contextmanager
def assert_failure(test, kind):
  """Checks a grouped lifecycle failure without losing its original type."""

  with test.assertRaises(BaseExceptionGroup) as caught:
    yield caught
  leaves = exception_leaves(caught.exception)
  test.assertTrue(any(isinstance(leaf, kind) for leaf in leaves))
