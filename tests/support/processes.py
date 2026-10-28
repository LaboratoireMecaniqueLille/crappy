"""Bounded waits and process ownership for tests and isolated probes."""

import os
import signal
import subprocess
import sys
from time import monotonic, sleep, time

import psutil

from tests.support.paths import PROJECT_ROOT


def wait_until(predicate, timeout: float = 3.0) -> bool:
  deadline = monotonic() + timeout
  while not predicate():
    remaining = deadline - monotonic()
    if remaining <= 0:
      return False
    sleep(min(0.005, remaining))
  return True


def wait_until_stable(get_value, stable_for=0.05, timeout=3.0):
  deadline = monotonic() + timeout
  last = get_value()
  unchanged_since = monotonic()
  while monotonic() < deadline:
    value, now = get_value(), monotonic()
    if value != last:
      last, unchanged_since = value, now
    elif now - unchanged_since >= stable_for:
      return value
    sleep(min(0.005, max(0, deadline - now)))
  raise AssertionError('shared value did not stabilize before its deadline')


def stop_process(process, request_stop, timeout: float = 3.0, drain=None) -> None:
  """Request normal shutdown; emergency termination must fail the test."""

  if process._closed or process.pid is None:
    return
  errors = []
  try:
    if process.is_alive():
      request_stop()
      deadline = monotonic() + timeout
      while process.is_alive() and monotonic() < deadline:
        if drain is not None:
          drain()
        process.join(min(0.02, max(0, deadline - monotonic())))
  except BaseException as error:
    errors.append(error)
  if process.is_alive():
    errors.append(AssertionError(f'{process.name} required forced termination'))
    try:
      process.terminate()
      process.join(1.0)
    except BaseException as error:
      errors.append(error)
    if process.is_alive():
      try:
        process.kill()
        process.join(1.0)
      except BaseException as error:
        errors.append(error)
  if errors:
    if len(errors) == 1:
      raise errors[0]
    raise BaseExceptionGroup('test process cleanup failed', errors)
  process.join(0)


def stop_thread(thread, request_stop, timeout: float = 3.0) -> None:
  """Stop a fixture-owned receiver before its shared buffers are released."""

  errors = []
  if thread.is_alive():
    try:
      request_stop()
    except BaseException as error:
      errors.append(error)
    thread.join(timeout)
  if thread.is_alive():
    errors.append(AssertionError(f'{thread.name} did not stop before its deadline'))
  if errors:
    raise BaseExceptionGroup('test thread cleanup failed', errors)


def _live(process):
  try:
    return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
  except psutil.NoSuchProcess:
    return False


def _owned_processes(process, known):
  try:
    parent = psutil.Process(process.pid)
    for child in parent.children(recursive=True):
      known[child.pid] = child
  except psutil.NoSuchProcess:
    pass
  # A POSIX session also retains descendants after the direct parent exits.
  if os.name == 'posix':
    for child in psutil.process_iter():
      try:
        if os.getpgid(child.pid) == process.pid and child.pid != process.pid:
          known[child.pid] = child
      except (ProcessLookupError, PermissionError):
        pass
  else:
    # Windows retains inherited parent IDs after a parent exits. Recover the
    # tree even if it exited between two polls, while rejecting older PIDs.
    pending = list(psutil.process_iter(['pid', 'ppid', 'create_time']))
    parents = {process.pid, *known}
    while pending:
      found = []
      for child in pending:
        info = child.info
        if (info['ppid'] in parents and info['create_time'] is not None
            and info['create_time'] >= process.test_started):
          known[child.pid] = child
          parents.add(child.pid)
          found.append(child)
      if not found:
        break
      pending = [child for child in pending if child not in found]
  return [child for child in known.values() if _live(child)]


def _terminate_tree(process, known):
  children = _owned_processes(process, known)
  if os.name == 'posix':
    try:
      os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
      pass
  else:
    for child in reversed(children):
      try:
        child.terminate()
      except psutil.NoSuchProcess:
        pass
    if process.poll() is None:
      process.terminate()
  _, alive = psutil.wait_procs(children, timeout=1.0)
  for child in alive:
    if _live(child):
      try:
        child.kill()
      except psutil.NoSuchProcess:
        pass
  if os.name == 'posix':
    # Include descendants created while a SIGTERM handler was running.
    try:
      os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
      pass
  if process.poll() is None:
    process.kill()
  process.wait(timeout=2.0)
  psutil.wait_procs(alive, timeout=1.0)
  survivors = _owned_processes(process, known)
  if survivors:
    raise AssertionError(f'subprocess descendants survived cleanup: {survivors}')


def run_python(arguments, *, timeout=45.0, env=None, cwd=PROJECT_ROOT,
               text=True, check=False):
  """Run the configured interpreter with captured output and a tree deadline.

  A successful parent leaving workers alive is a failed test. Timeout errors
  retain output collected while the owned process tree is stopped.
  """

  command = [sys.executable, *arguments]
  started = time()
  process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=text,
                             start_new_session=os.name == 'posix')
  process.test_started = started
  known = {}
  deadline = monotonic() + timeout
  try:
    while True:
      _owned_processes(process, known)
      remaining = deadline - monotonic()
      if remaining <= 0:
        raise subprocess.TimeoutExpired(command, timeout)
      try:
        stdout, stderr = process.communicate(timeout=min(0.1, remaining))
        break
      except subprocess.TimeoutExpired:
        continue
    descendants = _owned_processes(process, known)
    if descendants:
      psutil.wait_procs(descendants, timeout=0.5)
      descendants = _owned_processes(process, known)
    if descendants:
      _terminate_tree(process, known)
      raise AssertionError(f'subprocess exited with live descendants: {descendants}\n'
                           f'stdout:\n{stdout}\nstderr:\n{stderr}')
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check:
      result.check_returncode()
    return result
  except BaseException as error:
    try:
      _terminate_tree(process, known)
      stdout, stderr = process.communicate(timeout=2.0)
    except BaseException as cleanup_error:
      raise BaseExceptionGroup('subprocess failed and cleanup failed',
                               [error, cleanup_error]) from None
    if isinstance(error, subprocess.TimeoutExpired):
      error.output, error.stderr = stdout, stderr
      error.add_note(f'stdout:\n{stdout}\nstderr:\n{stderr}')
    raise
  finally:
    process.stdout.close()
    process.stderr.close()
