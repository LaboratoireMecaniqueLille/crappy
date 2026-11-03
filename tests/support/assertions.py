"""Failure setup and assertions shared by independent test fixtures."""

from typing import Any
from unittest.mock import Mock

from crappy._failure import FailureManager


def with_failure_manager(component: Any) -> Any:
  """Installs run-owned state for tests that call lifecycle hooks directly."""

  component._failure_manager = FailureManager(component.name, 'main')
  component._failure_manager._logger = Mock()
  return component


def exception_leaves(error: BaseException) -> tuple[BaseException, ...]:
  """Returns group leaves for assertions without traversing causes."""

  if isinstance(error, BaseExceptionGroup):
    return tuple(leaf for member in error.exceptions
                 for leaf in exception_leaves(member))
  return (error,)
