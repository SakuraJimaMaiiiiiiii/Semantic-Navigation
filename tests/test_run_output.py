"""每次运行独立数据目录的回归测试。"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from config import create_run_output_directory


class RunOutputTest(unittest.TestCase):
    def test_creates_timestamped_directory_under_requested_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "data" / "runs"

            output = create_run_output_directory(
                root,
                now=datetime(2026, 8, 25, 10, 20, 30, 456789),
            )

            self.assertEqual(
                output,
                root / "run_20260825_102030_456789",
            )
            self.assertTrue(output.is_dir())

            with self.assertRaises(FileExistsError):
                create_run_output_directory(
                    root,
                    now=datetime(2026, 8, 25, 10, 20, 30, 456789),
                )


if __name__ == "__main__":
    unittest.main()
