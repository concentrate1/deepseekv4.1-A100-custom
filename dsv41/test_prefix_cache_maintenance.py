"""Image-prefix cache expiry and exact-storage block compaction."""
import hashlib
import os
from pathlib import Path
import tempfile
import time
import unittest

import torch

from .prefix_cache_maintenance import clean_once, compact_block


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:32]


class PrefixCacheMaintenanceTest(unittest.TestCase):
    def test_compact_block_drops_parent_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = torch.arange(1024 * 64, dtype=torch.int64).view(1, 1024, 64)
            view = parent[:, :64]
            path = Path(tmp) / (digest(view) + ".pt")
            torch.save(view, path)
            before = path.stat().st_size
            self.assertGreater(before, view.numel() * view.element_size() * 2)
            self.assertTrue(compact_block(path))
            self.assertLess(path.stat().st_size, before / 2)
            self.assertTrue(torch.equal(torch.load(path, weights_only=False), view))

    def test_expire_old_manifest_and_unreferenced_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocks = root / "blocks"
            blocks.mkdir()
            old = torch.tensor([1], dtype=torch.int64)
            live = torch.tensor([2], dtype=torch.int64)
            old_hash, live_hash = digest(old), digest(live)
            old_block, live_block = blocks / (old_hash + ".pt"), blocks / (live_hash + ".pt")
            torch.save(old, old_block)
            torch.save(live, live_block)
            old_manifest = root / "prefix-old.pt"
            live_manifest = root / "prefix-live.pt"
            torch.save({"block_refs": [{"refs": [old_hash]}]}, old_manifest)
            torch.save({"block_refs": [{"refs": [live_hash]}]}, live_manifest)
            now = time.time()
            os.utime(old_manifest, (now - 25 * 3600, now - 25 * 3600))
            os.utime(old_block, (now - 2 * 3600, now - 2 * 3600))
            result = clean_once(root, now=now, max_age_s=24 * 3600,
                                block_grace_s=3600, compact=False)
            self.assertEqual(result["manifests_removed"], 1)
            self.assertEqual(result["blocks_removed"], 1)
            self.assertFalse(old_manifest.exists())
            self.assertFalse(old_block.exists())
            self.assertTrue(live_manifest.exists())
            self.assertTrue(live_block.exists())


if __name__ == "__main__":
    unittest.main()
