# coding: utf-8

from pathlib import Path
from re import findall
from runpy import run_path
import unittest
from urllib.parse import urlparse


class TestAiDiscovery(unittest.TestCase):
  """Keeps the hand-reviewed AI entry point published and internally valid."""

  _source_dir = (Path(__file__).resolve().parents[2] / 'docs' / 'source')
  _llms_path = _source_dir / 'llms.txt'

  def test_llms_file_is_published(self) -> None:
    """Checks that Sphinx copies llms.txt to the documentation root."""

    config = run_path(str(self._source_dir / 'conf.py'))
    self.assertIn('llms.txt', config['html_extra_path'])

  def test_llms_links_are_canonical_and_local_pages_exist(self) -> None:
    """Checks absolute links and the source pages behind documentation URLs."""

    source = self._llms_path.read_text(encoding='utf-8')
    links = findall(r'\[[^]]+\]\(([^)]+)\)', source)
    self.assertTrue(links)
    self.assertEqual(len(links), len(set(links)))

    documentation_prefix = '/en/latest/'
    for link in links:
      with self.subTest(link=link):
        parsed = urlparse(link)
        self.assertEqual(parsed.scheme, 'https')
        self.assertTrue(parsed.netloc)

        if parsed.netloc != 'crappy.readthedocs.io':
          continue
        self.assertTrue(parsed.path.startswith(documentation_prefix))
        relative = parsed.path.removeprefix(documentation_prefix)
        self.assertTrue(relative.endswith('.html'))
        page = self._source_dir / f'{relative[:-5]}.rst'
        self.assertTrue(page.is_file())

  def test_llms_file_states_the_supported_image_architectures(self) -> None:
    """Protects the architectural guidance required from text-only readers."""

    source = self._llms_path.read_text(encoding='utf-8')
    self.assertIn('VisionBlocks connected by ImageLinks are recommended',
                  source)
    self.assertIn('all-in-one Camera Blocks remain supported', source)
