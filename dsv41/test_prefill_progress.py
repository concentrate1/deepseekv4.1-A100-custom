"""Progress must follow completed model chunks, not elapsed-time guesses."""
import types
import threading
import time
import unittest

from .engine import Engine
from .stats import StatsTracker


class Event:
    def __init__(self, done):
        self.done = done

    def query(self):
        return self.done


class EngineView:
    def __init__(self, work, prompt_tokens=50726):
        self.work = work
        self.prompt_tokens = prompt_tokens

    def get_cache_stats(self):
        return {
            "slots": [{"slot_id": 0, "status": "prefilling", "prompt_tokens": self.prompt_tokens,
                       "start_time": time.perf_counter() - 20, "generated_tokens": 0}],
            "current_phase": "prefill", "prefill_work": self.work,
        }


class PrefillProgressTest(unittest.TestCase):
    def test_engine_counts_only_completed_cuda_events(self):
        state = {"events": [(2048, Event(True)), (4096, Event(True)),
                            (6144, Event(False))],
                 "total_tokens": 50726, "total_chunks": 25,
                 "start_pos": 0, "started_at": time.perf_counter() - 10}
        result = Engine._completed_prefill_work(state)
        self.assertEqual(result["completed_tokens"], 4096)
        self.assertEqual(result["completed_chunks"], 2)

    def test_block_replay_progress_uses_completed_suffix_tokens(self):
        state = {"completed_tokens": 75776, "total_tokens": 102325,
                 "completed_chunks": 148, "total_chunks": 200,
                 "start_pos": 61429, "started_at": time.perf_counter() - 318}
        work = Engine._completed_prefill_work(state)
        tracker = StatsTracker(sample_interval=3600)
        try:
            progress = tracker.get_metrics(
                EngineView(work, prompt_tokens=163754)
            )["prefill_progress"]
            self.assertTrue(progress["is_replay"])
            self.assertEqual(progress["reused_tokens"], 61429)
            self.assertEqual(progress["current_tokens"], 75776)
            self.assertEqual(progress["suffix_tokens"], 102325)
            self.assertEqual(progress["percent"], 74.1)
        finally:
            tracker.stop()

    def test_cold_guard_replay_does_not_become_a_cache_hit(self):
        engine = Engine.__new__(Engine)
        engine.model = types.SimpleNamespace(_prefill_progress=None)
        engine._slot_lock = threading.Lock()
        engine.slot_states = {1: {'status': 'prefilling', 'reused_tokens': 1, 'prefill_type': 'gpu_hit'}}
        engine._set_prefill_route(1301, 0, 'cold')
        self.assertEqual(engine.slot_states[1]['reused_tokens'], 0)
        engine._prefill_route['prefix_chunks'] = 2
        engine._prefix_replay_progress = dict(total_tokens=256, start_pos=1045,
            completed_tokens=128, completed_chunks=0, total_chunks=1, started_at=time.perf_counter()-1)
        work = engine._live_prefill_work()
        self.assertEqual(work['completed_tokens'], 1173)
        self.assertEqual(work['total_tokens'], 1301)
        self.assertEqual(work['reused_tokens'], 0)
        self.assertEqual(work['prefill_type'], 'cold')
        self.assertEqual(work['completed_chunks'], 2)
        self.assertEqual(work['total_chunks'], 3)
        tracker = StatsTracker(sample_interval=3600)
        try:
            progress = tracker.get_metrics(EngineView(work, 1301))['prefill_progress']
            self.assertFalse(progress['is_replay'])
            self.assertEqual(progress['percent'], 90.2)
        finally:
            tracker.stop()

    def test_no_gpu_progress_is_not_reported_as_measured_zero_percent(self):
        engine = Engine.__new__(Engine)
        engine.model = types.SimpleNamespace(_prefill_progress=None)
        engine._set_prefill_route(1000, 0, 'cold')
        self.assertIsNone(engine._live_prefill_work())

    def test_cache_hit_progress_spans_all_suffix_blocks(self):
        engine = Engine.__new__(Engine)
        engine.model = types.SimpleNamespace(_prefill_progress={
            'events': [(128, Event(True))], 'total_tokens': 512,
            'total_chunks': 4, 'start_pos': 1512, 'started_at': time.perf_counter()-1})
        engine._set_prefill_route(3048, 1000, 'host_replay')
        engine._prefix_replay_progress = dict(total_tokens=2048, start_pos=1000,
            completed_tokens=1024, completed_chunks=2, total_chunks=4, started_at=time.perf_counter()-5)
        work = engine._live_prefill_work()
        self.assertEqual(work['completed_tokens'], 1024)
        self.assertEqual(work['total_tokens'], 2048)
        self.assertEqual(work['reused_tokens'], 1000)
        self.assertEqual(work['prefill_type'], 'host_replay')

    def test_measured_route_overrides_speculative_slot_metadata(self):
        class StaleSlotView(EngineView):
            def get_cache_stats(self):
                stats = super().get_cache_stats()
                stats['slots'][0].update(reused_tokens=1200, prefill_type='gpu_hit')
                return stats
        tracker = StatsTracker(sample_interval=3600)
        try:
            work = dict(completed_tokens=500, total_tokens=2000, completed_chunks=1,
                        total_chunks=4, start_pos=0, elapsed_s=2,
                        reused_tokens=0, prefill_type='cold')
            p = tracker.get_metrics(StaleSlotView(work, 2000))['prefill_progress']
            self.assertFalse(p['is_replay'])
            self.assertEqual(p['prefill_type'], 'cold')
            self.assertEqual(p['reused_tokens'], 0)
        finally:
            tracker.stop()

    def test_cooperative_prefill_reports_real_recent_decode_rate(self):
        class MixedView:
            def get_cache_stats(self):
                return {'current_phase':'prefill','slots':[
                    {'slot_id':1,'status':'generating','tok_s':80},
                    {'slot_id':2,'status':'prefilling','prompt_tokens':9000}],
                    'decode_10s_tok_s':12.5,'combined_decode_tok_s':80}
        tracker=StatsTracker(sample_interval=3600)
        try:
            metrics=tracker.get_metrics(MixedView())
            self.assertEqual(metrics['throughput']['decode_10s_tok_s'],12.5)
        finally:
            tracker.stop()

    def test_dashboard_uses_measured_tokens_or_no_percentage(self):
        tracker = StatsTracker(sample_interval=3600)
        try:
            work = {"completed_tokens": 7168, "total_tokens": 50726,
                    "completed_chunks": 4, "total_chunks": 25,
                    "start_pos": 0, "elapsed_s": 8.0}
            measured = tracker.get_metrics(EngineView(work))["prefill_progress"]
            self.assertEqual(measured["current_tokens"], 7168)
            self.assertEqual(measured["percent"], 14.1)
            self.assertEqual(measured["speed_tok_s"], 896.0)
            self.assertIsNone(tracker.get_metrics(EngineView(None))["prefill_progress"])
        finally:
            tracker.stop()


if __name__ == "__main__":
    unittest.main()
