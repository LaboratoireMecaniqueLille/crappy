"""Explicit GUI group selection, before any optional toolkit is imported."""

import os
from unittest import SkipTest


def require_gui_group(group: str) -> None:
  """Keep each graphical backend in its own explicitly selected interpreter."""

  selected = os.environ.get('CRAPPY_TEST_GUI', '')
  if selected not in ('', 'tkinter', 'pyqt', 'display'):
    raise ValueError('CRAPPY_TEST_GUI must be tkinter, pyqt, display, or unset')
  if selected != group:
    raise SkipTest(f'Set CRAPPY_TEST_GUI={group} to run this graphical group')
