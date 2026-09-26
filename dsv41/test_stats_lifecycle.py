"""A completed POST must not remain in the active request count."""
import unittest

from .stats import StatsTracker


class StatsLifecycleTest(unittest.TestCase):
    def test_request_start_does_not_count_the_same_post_twice(self):
        stats = StatsTracker(sample_interval=3600)
        try:
            before = stats.get_metrics()["total_requests"]
            stats.client_connected()
            stats.record_request_start(42, stream=True)
            during = stats.get_metrics()
            self.assertEqual(during["active_clients"], 1)
            self.assertEqual(during["total_requests"], before + 1)
            stats.client_disconnected()
            self.assertEqual(stats.get_metrics()["active_clients"], 0)
        finally:
            stats.stop()

    def test_decode_chart_rejects_first_token_spikes_and_jev_rates(self):
        self.assertTrue(StatsTracker._valid_decode_rate(122.3))
        self.assertFalse(StatsTracker._valid_decode_rate(7452.3))
        stats = StatsTracker(sample_interval=3600)
        try:
            before = len(stats.tok_s_records)
            stats.record_throughput(7452.3, "chat")
            stats.record_throughput(250.0, "jev")
            self.assertEqual(len(stats.tok_s_records), before)
            stats.record_throughput(80.0, "chat")
            self.assertEqual(len(stats.tok_s_records), before + 1)
            self.assertEqual(stats.tok_s_records[-1][1], 80.0)
        finally:
            stats.stop()

    def test_live_chart_keeps_samples_when_full_history_resampling_changes(self):
        import collections
        from unittest.mock import patch

        stats = StatsTracker(sample_interval=3600)
        stats.stop()
        samples = [{"t": t, "tok_s": t % 73} for t in range(1000, 2001)]
        stats.history = collections.deque(samples)
        with patch("dsv41.stats.time.time", return_value=2000):
            first = stats.get_metrics()
        stats.history.append({"t": 2005, "tok_s": 50})
        with patch("dsv41.stats.time.time", return_value=2005):
            second = stats.get_metrics()
        self.assertEqual(first["chart_time"], 2000)
        self.assertEqual(first["throughput_series"], samples[399:])
        self.assertEqual(second["throughput_series"][:-1], samples[404:])
        # Both windows retain identical timestamps and values in their overlap.
        self.assertEqual(first["throughput_series"][5:], second["throughput_series"][:-1])
        self.assertLessEqual(len(second["series"]), 300)

    def test_input_image_thumbnail_keeps_block_order(self):
        import base64
        import io
        from PIL import Image

        image = Image.new("RGB", (300, 100), (255, 0, 0))
        output = io.BytesIO()
        image.save(output, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "before"},
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "after"},
        ]}]}
        blocks = StatsTracker._input_messages(body)[0]["blocks"]
        tracker = StatsTracker(sample_interval=3600)
        try:
            tracker.record_input(body, "chat/completions", 100, "image-test")
            self.assertTrue(tracker.get_metrics()["recent_inputs"][-1]["has_image"])
        finally:
            tracker.stop()
        self.assertEqual([item["type"] for item in blocks], ["text", "image", "text"])
        self.assertEqual(blocks[0]["text"], "before")
        self.assertEqual(blocks[2]["text"], "after")
        thumbnail = base64.b64decode(blocks[1]["src"].split(",", 1)[1])
        with Image.open(io.BytesIO(thumbnail)) as result:
            self.assertLessEqual(max(result.size), 256)

    def test_input_preview_keeps_text_and_hides_image_payload(self):
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}},
        ]}]}
        preview, truncated = StatsTracker._summarize_input(body)
        self.assertIn("user: What is in this image?", preview)
        self.assertIn("[image]", preview)
        self.assertNotIn("SECRET", preview)
        self.assertFalse(truncated)
        long_preview, truncated = StatsTracker._summarize_input({"prompt": "a" * 7000})
        self.assertTrue(truncated)
        self.assertTrue(long_preview.endswith("a" * 6000))


if __name__ == "__main__":
    unittest.main()
