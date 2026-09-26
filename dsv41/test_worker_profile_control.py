"""Private profiling control must not interrupt active generation."""
import threading
import types
import unittest
import torch

from .model_worker import _route_summary, _serve_connection


class Connection:
    def __init__(self, req):
        self.req = req
        self.sent = []
    def recv(self):
        return self.req
    def send(self, value):
        self.sent.append(value)
    def close(self):
        pass


class WorkerProfileControlTests(unittest.TestCase):
    def test_route_summary_counts_unique_experts_without_exporting_ids(self):
        eid = torch.tensor([[[0, 1, 2], [0, 3, 4]],
                            [[4, 4, 4], [1, 1, 0]]])
        rt = types.SimpleNamespace(route_log=True,
                                   shard={"cuda:0": (0, 3), "cuda:1": (3, 2)},
                                   route_snapshot=lambda: (eid, None))
        result = _route_summary(rt)
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["mean_unique"], 4)
        self.assertEqual(result["layers"][0]["by_shard"], {"cuda:0": 3, "cuda:1": 2})
        self.assertEqual(result["layers"][1]["by_shard"], {"cuda:0": 2, "cuda:1": 1})

    def test_ep_trace_is_read_only_and_refuses_busy_worker(self):
        engine = types.SimpleNamespace(rt=types.SimpleNamespace(trace=None))
        lock = threading.Lock()
        req = {"protocol": 1, "operation": "ep_trace"}
        lock.acquire()
        busy = Connection(req)
        _serve_connection(busy, engine, lock)
        self.assertEqual(busy.sent[0]["kind"], "error")
        lock.release()
        idle = Connection(req)
        _serve_connection(idle, engine, lock)
        self.assertEqual(idle.sent[0]["kind"], "result")
        self.assertEqual(idle.sent[0]["value"], "no trace")

    def test_profile_can_toggle_only_when_idle(self):
        engine = types.SimpleNamespace(phase_profile_enabled=False,
                                       last_phase_profile={"steps": 1})
        lock = threading.Lock()
        req = {"protocol": 1, "operation": "set_phase_profile", "enabled": True}
        lock.acquire()
        busy = Connection(req)
        _serve_connection(busy, engine, lock)
        self.assertEqual(busy.sent[0]["kind"], "error")
        self.assertFalse(engine.phase_profile_enabled)
        lock.release()
        idle = Connection(req)
        _serve_connection(idle, engine, lock)
        self.assertEqual(idle.sent[0]["kind"], "result")
        self.assertTrue(engine.phase_profile_enabled)
        self.assertIsNone(engine.last_phase_profile)


if __name__ == "__main__":
    unittest.main()
