"""Image-aware prefix keys and safe anchor routing without model weights."""
import os
import tempfile
import time
from pathlib import Path
import threading
import types
import unittest
from unittest.mock import Mock

import torch

from .engine import Engine
from .model import SharedAttn as RuntimeSharedAttn
from .vision import ImageInput


class ImagePrefixCacheTest(unittest.TestCase):
    def test_pixel_and_token_type_identity_prevents_false_hits(self):
        image = ImageInput(2, torch.zeros((1, 3, 14, 14), dtype=torch.bfloat16),
                           1, 1, torch.tensor([0, 1, 3]))
        types = torch.tensor([[-1, -1, 0, 1, 3, -1]])
        key, end = Engine._image_cache_identity([[image]], types)
        self.assertEqual(end, 5)
        self.assertEqual(key, Engine._image_cache_identity([[image]], types)[0])
        changed_pixels = ImageInput(2, torch.ones_like(image.patches), 1, 1, image.types)
        self.assertNotEqual(key, Engine._image_cache_identity([[changed_pixels]], types)[0])
        changed_types = types.clone()
        changed_types[0, 3] = 2
        self.assertNotEqual(key, Engine._image_cache_identity([[image]], changed_types)[0])
        self.assertNotEqual(Engine._prefix_tmpfs_hash([1, 2, 3], key),
                            Engine._prefix_tmpfs_hash([1, 2, 3], "another-image"))

    def test_text_and_different_images_cannot_match_same_token_ids(self):
        engine = object.__new__(Engine)
        ids = [10, 129264, 129264, 11]
        entries = [
            {"base_ids": ids[:3], "image_key": None},
            {"base_ids": ids[:3], "image_key": "image-a"},
        ]
        engine._prefix_cache_entries = lambda: entries
        self.assertIs(engine._find_prefix_cache_entry(ids), entries[0])
        self.assertIs(engine._find_prefix_cache_entry(ids, "image-a"), entries[1])
        self.assertIsNone(engine._find_prefix_cache_entry(ids, "image-b"))

    def _fake_engine(self, ids):
        engine = object.__new__(Engine)
        seen = {}
        def forward(tokens, start_pos, images=None, token_types=None):
            seen.update(tokens=tokens, start_pos=start_pos,
                        images=images, token_types=token_types)
            return torch.zeros((1, 10))
        engine.model = types.SimpleNamespace(
            blocks=[types.SimpleNamespace(device=torch.device("cpu"))],
            forward=forward,
        )
        engine.ds = None
        engine._snapshot_prefix_state = Mock(return_value=(["snapshot"], 100))
        engine._replay_prefix_tail = Mock(return_value=(torch.ones((1, 10)), None))
        engine._store_prefix_cache_entry = Mock()
        engine._record_prefill_stats = Mock()
        engine._image_cache_identity = Mock(return_value=("image-a", 5))
        engine._find_prefix_cache_entry = Mock(return_value=None)
        engine._prefix_prompt_ids = [1]
        engine._prefix_len = 1
        return engine, seen

    def test_cold_anchor_covers_image_and_replays_only_text(self):
        ids = list(range(100))
        engine, seen = self._fake_engine(ids)
        images = [[object()]]
        types = torch.full((1, len(ids)), -1, dtype=torch.long)
        logits, reused = engine._prefill_with_image_cache(ids, images, types)
        self.assertEqual(reused, 0)
        self.assertEqual(tuple(logits.shape), (1, 10))
        self.assertEqual(seen["tokens"].shape[1], 5)
        self.assertIs(seen["images"], images)
        self.assertEqual(seen["token_types"].shape[1], 5)
        engine._replay_prefix_tail.assert_called_once_with(ids, 5, snapshot_at=None)
        self.assertEqual(engine._store_prefix_cache_entry.call_args.kwargs["image_key"], "image-a")
        self.assertEqual(engine._store_prefix_cache_entry.call_args.kwargs["base_ids"], ids[:5])

    def test_hit_replays_text_suffix_without_vision_forward(self):
        ids = list(range(100))
        engine, seen = self._fake_engine(ids)
        entry = {"base_ids": ids[:80], "prompt_ids": ids[:90],
                 "snapshot": ["saved"], "bytes": 100, "image_key": "image-a"}
        engine._find_prefix_cache_entry.return_value = entry
        engine._restore_prefix_state = Mock()
        logits, reused = engine._prefill_with_image_cache(
            ids, [[object()]], torch.full((1, 100), -1))
        self.assertEqual(reused, 80)
        self.assertEqual(seen, {})
        engine._restore_prefix_state.assert_called_once_with(["saved"])
        engine._replay_prefix_tail.assert_called_once_with(ids, 80, snapshot_at=None)
        self.assertEqual(engine._store_prefix_cache_entry.call_args.kwargs["image_key"], "image-a")

    def test_image_at_end_or_disabled_does_not_snapshot(self):
        ids = list(range(100))
        engine, seen = self._fake_engine(ids)
        engine._image_cache_identity.return_value = ("image-a", 100)
        engine._prefill_with_image_cache(ids, [[object()]], torch.full((1, 100), -1))
        self.assertEqual(seen["tokens"].shape[1], 100)
        engine._snapshot_prefix_state.assert_not_called()
        engine2, seen2 = self._fake_engine(ids)
        engine2._prefill_with_image_cache(ids, [[object()]], torch.full((1, 100), -1),
                                          cache_disabled=True)
        self.assertEqual(seen2["tokens"].shape[1], 100)
        engine2._snapshot_prefix_state.assert_not_called()

    def test_image_identity_survives_disk_snapshot_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("DSV41_IMAGE_PREFIX_CACHE_DIR")
            os.environ["DSV41_IMAGE_PREFIX_CACHE_DIR"] = tmp
            try:
                holder = types.SimpleNamespace(window_kv_cache=torch.zeros((1, 4, 2)))
                slot = ("attr", holder, "window_kv_cache", torch.ones((1, 4, 2)))
                source = object.__new__(Engine)
                source.max_seq_len = 1024
                source.tok = types.SimpleNamespace(name_or_path="test-model")
                entry = {"prompt_ids": [1, 2, 3, 4], "base_ids": [1, 2, 3],
                         "image_key": "pixel-digest", "snapshot": [slot], "bytes": 32}
                source._persist_prefix_cache_entry(entry)
                self.assertEqual(len(list(Path(tmp).glob("prefix-*.pt"))), 1)

                holder2 = types.SimpleNamespace(window_kv_cache=torch.zeros((1, 4, 2)))
                current_slot = ("attr", holder2, "window_kv_cache", holder2.window_kv_cache)
                target = object.__new__(Engine)
                target.max_seq_len = 1024
                target.tok = types.SimpleNamespace(name_or_path="test-model")
                target._prefix_current_slot_signatures = lambda: (
                    [current_slot], [target._prefix_slot_signature(current_slot)],
                )
                loaded = []
                target._load_prefix_cache_from_tmpfs(loaded, root=tmp)
                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded[0]["image_key"], "pixel-digest")
                self.assertEqual(loaded[0]["base_ids"], [1, 2, 3])
                self.assertLessEqual(loaded[0]["last_used"], time.monotonic())
                target._cached_checkpoint_fingerprint = "different-checkpoint"
                rejected = []
                target._load_prefix_cache_from_tmpfs(rejected, root=tmp)
                self.assertEqual(rejected, [])
            finally:
                if previous is None:
                    os.environ.pop("DSV41_IMAGE_PREFIX_CACHE_DIR", None)
                else:
                    os.environ["DSV41_IMAGE_PREFIX_CACHE_DIR"] = previous

    def test_large_index_snapshot_grows_index_table_on_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("DSV41_PREFIX_CACHE_DIR")
            os.environ["DSV41_PREFIX_CACHE_DIR"] = tmp
            try:
                key = (2, "cpu")
                source = object.__new__(Engine)
                source.max_seq_len = 1024
                source.tok = types.SimpleNamespace(name_or_path="test-model")
                index = {key: torch.ones((1, 4, 2))}
                source._persist_prefix_cache_entry({
                    "prompt_ids": [1, 2, 3, 4, 5], "base_ids": [1, 2, 3, 4],
                    "snapshot": [("dict", index, key, index[key])], "bytes": 32,
                })

                shared = RuntimeSharedAttn()
                shared.cache_max_rows[2] = 8
                shared.index_k[key] = torch.zeros((1, 2, 2))
                shared.compress_kv[key] = torch.zeros((1, 2, 2))
                target = object.__new__(Engine)
                target.max_seq_len = 1024
                target.tok = types.SimpleNamespace(name_or_path="test-model")
                target.model = types.SimpleNamespace(shared=shared)
                slot = ("dict", shared.index_k, key, shared.index_k[key])
                target._prefix_current_slot_signatures = lambda: (
                    [slot], [target._prefix_slot_signature(slot)],
                )
                loaded = []
                target._load_prefix_cache_from_tmpfs(loaded, root=tmp)
                self.assertEqual(len(loaded), 1)
                self.assertGreaterEqual(shared.index_k[key].shape[1], 4)
                self.assertEqual(shared.compress_kv[key].shape[1], 2)
                self.assertTrue(torch.equal(loaded[0]["snapshot"][0][3], index[key]))
            finally:
                if old is None:
                    os.environ.pop("DSV41_PREFIX_CACHE_DIR", None)
                else:
                    os.environ["DSV41_PREFIX_CACHE_DIR"] = old

    def test_text_lru_does_not_evict_image_family(self):
        from unittest.mock import patch
        engine = object.__new__(Engine)
        text_old = {"base_ids": [1], "prompt_ids": [1, 2], "image_key": None,
                    "bytes": 1, "last_used": 1.0, "snapshot": ["old"]}
        image_old = {"base_ids": [3], "prompt_ids": [3, 4], "image_key": "image-a",
                     "bytes": 1, "last_used": 2.0, "snapshot": ["image"]}
        entries = [text_old, image_old]
        engine._prefix_cache_entries = lambda: entries
        engine._persist_prefix_cache_entry = Mock()
        engine.stats_tracker = None
        with patch.dict(os.environ, {"DSV41_PREFIX_CACHE_ENTRIES": "1",
                                     "DSV41_IMAGE_PREFIX_CACHE_ENTRIES": "2"}):
            engine._store_prefix_cache_entry(
                prompt_ids=[5, 6], base_ids=[5], snapshot=["new"],
                snapshot_bytes=1, image_key=None,
            )
        self.assertNotIn(text_old, entries)
        self.assertIn(image_old, entries)
        self.assertEqual(len([e for e in entries if e.get("image_key")]), 1)

    def test_same_base_refresh_uses_monotonic_lru(self):
        engine = object.__new__(Engine)
        previous = {"base_ids": [1, 2], "prompt_ids": [1, 2, 3],
                    "image_key": "image-a", "last_used": 0.0, "hits": 0}
        result = engine._store_prefix_cache_entry(
            prompt_ids=[1, 2, 3, 4], base_ids=[1, 2],
            snapshot=["snapshot"], snapshot_bytes=1,
            previous=previous, image_key="image-a",
        )
        self.assertIs(result, previous)
        self.assertLess(abs(previous["last_used"] - time.monotonic()), 1.0)

    def test_image_request_invalidates_gpu_token_id_signature(self):
        engine = object.__new__(Engine)
        engine.model = types.SimpleNamespace(_prefill_progress=None)
        engine._slot_tokens_lock = threading.Lock()
        engine._slot_tokens = {0: [1, 2, 3]}
        engine._prefix_prompt_ids = [1, 2, 3]
        engine._prefix_len = 3
        engine._disable_prefix_cache_once = False
        engine._prefill_with_image_cache = Mock(return_value=(torch.zeros((1, 10)), 0))
        engine._prefill_with_prefix_reuse([1, 129264, 2], [[object()]], torch.tensor([[-1, 1, -1]]))
        self.assertEqual(engine._slot_tokens[0], [-1])
        self.assertIsNone(engine._prefix_prompt_ids)
        self.assertEqual(engine._prefix_len, 0)


if __name__ == "__main__":
    unittest.main()
