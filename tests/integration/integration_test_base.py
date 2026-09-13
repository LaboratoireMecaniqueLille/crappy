# coding: utf-8

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Iterator
import unittest

try:
  import psutil
except (ImportError, ModuleNotFoundError):
  psutil = None


class IntegrationTestBase(unittest.TestCase):
  """Base class for running isolated Crappy integration scenarios."""

  scenario_timeout = 15.0
  _project_root = Path(__file__).resolve().parents[2]

  @contextmanager
  def run_scenario(self,
                   scenario: str,
                   timeout: float | None = None) -> Iterator[Path]:
    """Runs one scenario and yields its temporary artifact directory."""

    with TemporaryDirectory(prefix=f'crappy-{scenario}-') as folder:
      output_dir = Path(folder)
      stdout, stderr = self._execute_scenario(
        scenario,
        output_dir,
        self.scenario_timeout if timeout is None else timeout)

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
                        timeout: float) -> tuple[str, str]:
    """Runs a scenario subprocess and requires successful completion."""

    returncode, stdout, stderr = self._run_scenario_process(
      scenario, output_dir, timeout)

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
                            timeout: float) -> tuple[int, str, str]:
    """Starts a scenario subprocess and enforces its hard timeout."""

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    process = subprocess.Popen(
      [sys.executable, '-m', 'tests.integration.run_scenario',
       scenario, str(output_dir)],
      cwd=self._project_root,
      env=env,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True)

    try:
      stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
      self._terminate_process_tree(process)
      stdout, stderr = process.communicate()
      self.fail(self._failure_message(
        scenario,
        stdout,
        stderr,
        f'The scenario exceeded its {timeout:g}s timeout.'))

    if process.returncode is None:
      raise RuntimeError("The completed scenario has no return code")

    return process.returncode, stdout, stderr

  @staticmethod
  def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Terminates a timed-out scenario and all its child processes."""

    if psutil is None:
      process.kill()
      return

    try:
      parent = psutil.Process(process.pid)
      processes = parent.children(recursive=True)
      processes.append(parent)
    except psutil.NoSuchProcess:
      return

    for child in processes:
      try:
        child.terminate()
      except psutil.NoSuchProcess:
        pass

    _, alive = psutil.wait_procs(processes, timeout=2.0)
    for child in alive:
      try:
        child.kill()
      except psutil.NoSuchProcess:
        pass

    psutil.wait_procs(alive, timeout=2.0)

  @staticmethod
  def _failure_message(scenario: str,
                       stdout: str,
                       stderr: str,
                       reason: str) -> str:
    """Formats subprocess diagnostics for an integration-test failure."""

    return (f'{reason}\nScenario: {scenario}\n'
            f'----- stdout -----\n{stdout}\n'
            f'----- stderr -----\n{stderr}')
