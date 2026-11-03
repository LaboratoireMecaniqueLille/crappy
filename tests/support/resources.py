"""Small cleanup and state-restoration helpers for unittest fixtures."""

import logging
from queue import Empty
from time import monotonic


def run_cleanups(*callbacks) -> None:
  """Attempt every independent release and retain all failures."""

  errors = []
  for callback in callbacks:
    try:
      callback()
    except BaseException as error:
      errors.append(error)
  if errors:
    raise BaseExceptionGroup('test resource cleanup failed', errors)


def preserve_mapping(test, mapping) -> None:
  """Restore the contents of an existing registry after this test."""

  previous = mapping.copy()

  def restore():
    mapping.clear()
    mapping.update(previous)

  test.addCleanup(restore)


def isolate_logger(test, name: str, level: int | None = None):
  """Temporarily detach handlers without closing somebody else's handlers."""

  logger = logging.getLogger(name)
  previous = (logger.handlers[:], logger.level, logger.disabled, logger.propagate,
              logging.root.manager.disable)

  def restore():
    handlers, old_level, disabled, propagate, global_disable = previous
    added = [handler for handler in logger.handlers if handler not in handlers]
    logger.handlers = handlers
    logger.setLevel(old_level)
    logger.disabled, logger.propagate = disabled, propagate
    logging.disable(global_disable)
    run_cleanups(*(handler.close for handler in added))

  test.addCleanup(restore)
  logger.handlers = []
  logger.disabled = False
  logger.propagate = False
  if level is not None:
    logger.setLevel(level)
  return logger


def close_queue(queue, timeout: float = 1.0) -> None:
  """Drain a fixture-owned queue before waiting for its feeder to finish."""

  # Queue.join_thread has no timeout. Observe its actual feeder with a bounded
  # join instead, and use cancel_join_thread only after reporting a leak.
  deadline = monotonic() + timeout
  if not queue._closed:
    while monotonic() < deadline:
      try:
        queue.get(timeout=min(0.01, max(0, deadline - monotonic())))
      except Empty:
        break
  queue.close()
  feeder = queue._thread
  if feeder is not None:
    feeder.join(max(0, deadline - monotonic()))
    if feeder.is_alive():
      queue.cancel_join_thread()
      raise AssertionError('a test queue feeder did not stop before its deadline')


def drain_queue(queue) -> None:
  """Discard teardown-only output while its producer is joining."""

  deadline = monotonic() + 0.05
  if not queue._closed:
    try:
      while monotonic() < deadline:
        queue.get_nowait()
    except Empty:
      pass
