"""CI discovery cannot succeed without executing a healthy test group."""

from contextlib import redirect_stderr
from io import StringIO
import unittest

from tests.run import run_suite


class TestCIRunner(unittest.TestCase):
  @staticmethod
  def run_group(*tests) -> int:
    """Observe a nested suite's status without printing expected failures."""

    with redirect_stderr(StringIO()):
      return run_suite(unittest.TestSuite(tests))

  def test_empty_collection_fails(self):
    self.assertEqual(self.run_group(), 1)

  def test_entirely_skipped_group_fails(self):
    def skip():
      raise unittest.SkipTest('required capability is absent')

    self.assertEqual(self.run_group(unittest.FunctionTestCase(skip)), 1)

  def test_executed_tests_allow_local_capability_skips(self):
    def skip():
      raise unittest.SkipTest('optional local capability is absent')

    self.assertEqual(self.run_group(unittest.FunctionTestCase(lambda: None),
                                    unittest.FunctionTestCase(skip)), 0)

  def test_skipped_class_setup_fails_the_group(self):
    class UnavailableBackend(unittest.TestCase):
      @classmethod
      def setUpClass(cls):
        raise unittest.SkipTest('backend is unavailable')

      def test_window(self):
        pass

    self.assertEqual(self.run_group(UnavailableBackend('test_window')), 1)

  def test_skipped_subtests_do_not_hide_an_executed_test(self):
    class PartialCapabilities(unittest.TestCase):
      def test_capabilities(self):
        for available in (True, False, False):
          with self.subTest(available=available):
            if not available:
              self.skipTest('optional subcase capability is absent')
            self.assertTrue(available)

    case = PartialCapabilities('test_capabilities')
    self.assertEqual(self.run_group(case), 0)

  def test_collection_errors_fail_the_group(self):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('_crappy_missing_ci_test_module')
    self.assertTrue(loader.errors)
    with redirect_stderr(StringIO()):
      self.assertEqual(run_suite(suite), 1)
