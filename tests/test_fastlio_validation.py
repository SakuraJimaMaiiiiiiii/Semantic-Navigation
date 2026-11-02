"""Validation recovery must preserve source bytes and reject interior corruption."""
import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from tools.prepare_fastlio_validation import prepare, digest


class ValidationCopyTests(unittest.TestCase):
    def make_input(self, path, stamps):
        data = np.ones((len(stamps), 10))
        data[:, 0] = stamps
        data[np.asarray(stamps) == 0] = 0
        with h5py.File(path, 'w') as f:
            f.create_dataset('imu', data=data, maxshape=(None, 10))
            f.create_group('scans').create_dataset('sentinel', data=[4, 5, 6])

    def test_only_zero_suffix_removed_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'inputs.h5';out=Path(tmp)/'copy'
            self.make_input(source, [1., 1.01, 1.02, 0., 0.])
            before=digest(source)
            prepare(source,out)
            self.assertEqual(digest(source),before)
            with h5py.File(out/'inputs.h5') as f:
                np.testing.assert_array_equal(f['imu'][:,0],[1.,1.01,1.02])
                np.testing.assert_array_equal(f['scans/sentinel'][:],[4,5,6])
            report=json.loads((out/'recovery_report.json').read_text())
            self.assertEqual(report['removed_trailing_zero_imu_rows'],2)
            self.assertFalse(report['original_capture_complete'])
            with self.assertRaises(FileExistsError):
                prepare(source,out)

    def test_interior_corruption_is_not_sorted_or_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'inputs.h5';out=Path(tmp)/'copy'
            self.make_input(source,[1.,0.,2.,0.])
            with self.assertRaisesRegex(ValueError,'invalid'):
                prepare(source,out)
            self.assertFalse(out.exists())
