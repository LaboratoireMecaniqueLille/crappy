# coding: utf-8

from tests.support.paths import PROJECT_ROOT

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Iterator
import unittest

from tests.support.processes import run_python


class IntegrationTestBase(unittest.TestCase):
  """Base class for running isolated Crappy integration scenarios."""

  scenario_timeout = 15.0
  _project_root = PROJECT_ROOT

  @contextmanager
  def run_scenario(self,
                   scenario: str,
                   timeout: float | None = None,
                   start_method: str | None = None) -> Iterator[Path]:
    """Runs one scenario and yields its temporary artifact directory."""

    with TemporaryDirectory(prefix=f'crappy-{scenario}-') as folder:
      output_dir = Path(folder)
      stdout, stderr = self._execute_scenario(
        scenario,
        output_dir,
        self.scenario_timeout if timeout is None else timeout,
        start_method)

      self.assertNotIn(
        'leaked shared_memory objects',
        stderr,
        self._failure_message(
          scenario, stdout, stderr,
          'The scenario leaked a tracked shared-memory object.'))

      completion_path = output_dir / 'completed.json'
      self.assertTrue(
        completion_path.is_file(),
        self._failure_message(scenario, stdout, stderr,
                              'The completion marker was not generated.'))

      with completion_path.open(encoding='utf-8') as completion_file:
        completion = json.load(completion_file)

      self.assertEqual(completion, {'scenario': scenario})
      yield output_dir

  def _execute_scenario(self,
                        scenario: str,
                        output_dir: Path,
                        timeout: float,
                        start_method: str | None = None) -> tuple[str, str]:
    """Runs a scenario subprocess and requires successful completion."""

    returncode, stdout, stderr = self._run_scenario_process(
      scenario, output_dir, timeout, start_method)

    self.assertEqual(
      returncode,
      0,
      self._failure_message(
        scenario,
        stdout,
        stderr,
        f'The scenario exited with code {returncode}.'))

    return stdout, stderr

  def assert_scenario_fails(self,
                            scenario: str,
                            expected_error: str,
                            timeout: float | None = None) -> None:
    """Checks that a scenario fails promptly with the expected diagnostic."""

    with TemporaryDirectory(prefix=f'crappy-{scenario}-') as folder:
      output_dir = Path(folder)
      returncode, stdout, stderr = self._run_scenario_process(
        scenario,
        output_dir,
        self.scenario_timeout if timeout is None else timeout)

      self.assertNotEqual(
        returncode,
        0,
        self._failure_message(
          scenario, stdout, stderr,
          'The scenario unexpectedly exited successfully.'))
      self.assertFalse(
        (output_dir / 'completed.json').exists(),
        self._failure_message(
          scenario, stdout, stderr,
          'A failing scenario generated a completion marker.'))
      self.assertIn(
        expected_error,
        f'{stdout}\n{stderr}',
        self._failure_message(
          scenario, stdout, stderr,
          f'The expected diagnostic {expected_error!r} was not emitted.'))

  def _run_scenario_process(self,
                            scenario: str,
                            output_dir: Path,
                            timeout: float,
                            start_method: str | None = None
                            ) -> tuple[int, str, str]:
    """Run with captured diagnostics and bounded cleanup of the child tree."""
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    arguments = ['-m', 'tests.integration.run_scenario', scenario, str(output_dir)]
    if start_method is not None:
      arguments.extend(('--start-method', start_method))
    try:
      result = run_python(arguments, timeout=timeout, env=env, cwd=self._project_root)
    except subprocess.TimeoutExpired as error:
      self.fail(self._failure_message(
        scenario, error.stdout, error.stderr,
        f'The scenario exceeded its {timeout:g}s timeout.'))
    return result.returncode, result.stdout, result.stderr

  @staticmethod
  def _failure_message(scenario: str,
                       stdout: str,
                       stderr: str,
                       reason: str) -> str:
    """Formats subprocess diagnostics for an integration-test failure."""

    return (f'{reason}\nScenario: {scenario}\n'
            f'----- stdout -----\n{stdout}\n'
            f'----- stderr -----\n{stderr}')
