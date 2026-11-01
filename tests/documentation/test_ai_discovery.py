# coding: utf-8

from tests.support.paths import PROJECT_ROOT

from re import findall
from runpy import run_path
import unittest
from urllib.parse import urlparse


class TestAiDiscovery(unittest.TestCase):
  """Keeps the hand-reviewed AI entry point published and internally valid."""

  _source_dir = (PROJECT_ROOT / 'docs' / 'source')
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

  def test_llms_file_links_to_supported_image_architectures(self) -> None:
    """Readers can reach the supported image architecture comparison."""

    source = self._llms_path.read_text(encoding='utf-8')
    links = findall(r'\[[^]]+\]\(([^)]+)\)', source)
    target = 'concepts/image_pipelines.html'
    self.assertTrue(any(urlparse(link).path.endswith('/' + target)
                        for link in links))
    self.assertTrue((self._source_dir / 'concepts' / 'image_pipelines.rst').is_file())
