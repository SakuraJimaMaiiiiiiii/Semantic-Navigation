"""Isolate model-stack imports from the remaining SDK/native unit tests."""
import subprocess
import sys
import unittest


class ExploreV3IntegrationTests(unittest.TestCase):
    def test_backend_compatibility_in_clean_process(self):
        for pattern in ('explore_v3_cases.py', 'explore_v3_replay_cases.py'):
            with self.subTest(pattern=pattern):
                result = subprocess.run([
                    sys.executable, '-X', 'utf8', '-m', 'unittest', 'discover',
                    '-s', 'tests/backend_cases', '-p', pattern,
                ], capture_output=True, text=True, encoding='utf-8', timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
