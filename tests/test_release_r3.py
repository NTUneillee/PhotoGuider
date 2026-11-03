"""CPU-only tests for asset verification and reproduction comparisons."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

RELEASE = Path(__file__).resolve().parents[1]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class ReleaseTests(unittest.TestCase):
    def test_assets_missing_present_corrupt(self):
        mod = module('assets', RELEASE/'envs/verify_assets.py')
        with tempfile.TemporaryDirectory() as tmp:
            mod.RELEASE = Path(tmp)
            mod.pins = lambda: {'files': {'model': hashlib.sha256(b'valid').hexdigest()}}
            mod.verify()
            with self.assertRaises(FileNotFoundError):
                mod.verify(require_present=True)
            (mod.RELEASE/'model').write_bytes(b'valid')
            mod.verify(require_present=True)
            (mod.RELEASE/'model').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                mod.verify()

    def test_numeric_comparison_tolerance(self):
        mod = module('integrity', RELEASE/'tools/check_results_integrity.py')
        mod.EXPECT = {'test': None}
        with tempfile.TemporaryDirectory() as tmp:
            reference, actual = Path(tmp)/'reference', Path(tmp)/'actual'
            reference.mkdir(); actual.mkdir(); mod.RESULTS = reference
            (reference/'boxes_test.csv').write_text('id,bde\n1,0.1\n')
            (actual/'boxes_test.csv').write_text('id,bde\n1,0.1000000000000001\n')
            mod.compare_reproduced(actual)
            (actual/'boxes_test.csv').write_text('id,bde\n1,0.10000001\n')
            with self.assertRaises(AssertionError):
                mod.compare_reproduced(actual)

if __name__ == '__main__':
    unittest.main()
