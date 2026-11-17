"""Cache correctness: original history, branch isolation, crop, and bounded storage. CPU only."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import torch
from transformers.cache_utils import DynamicLayer
from vega_stream import _ReservedKV


class KVCacheTests(unittest.TestCase):
    def test_append_crop_and_branches(self):
        ref, fast = DynamicLayer(), _ReservedKV(24)
        pointer = None
        torch.manual_seed(7)
        for length in (3, 5, 4):
            k, v = [torch.randn(1, 2, length, 8) for _ in range(2)]
            expected, actual = ref.update(k, v), fast.update(k, v)
            for a, b in zip(expected, actual):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(ref.get_seq_length(), fast.get_seq_length())
            self.assertEqual(ref.get_mask_sizes(5), fast.get_mask_sizes(5))
            current = fast._keys_storage.untyped_storage().data_ptr()
            self.assertTrue(pointer is None or pointer == current)
            pointer = current
        branch = copy.deepcopy(fast)
        self.assertNotEqual(branch._keys_storage.untyped_storage().data_ptr(), pointer)
        self.assertEqual(branch.keys.untyped_storage().data_ptr(), branch._keys_storage.untyped_storage().data_ptr())
        branch.crop(-2)
        k, v = [torch.randn(1, 2, 1, 8) for _ in range(2)]
        branch.update(k, v)
        self.assertEqual(branch.get_seq_length(), 11)
        self.assertEqual(fast.get_seq_length(), 12)
        self.assertTrue(torch.equal(fast.keys, ref.keys), "one question must not modify another's prefix")
        before = fast.keys.clone()
        with self.assertRaisesRegex(ValueError, "capacity"):
            fast.update(torch.zeros(1, 2, 20, 8), torch.zeros(1, 2, 20, 8))
        self.assertTrue(torch.equal(fast.keys, before))


if __name__ == "__main__":
    unittest.main()
