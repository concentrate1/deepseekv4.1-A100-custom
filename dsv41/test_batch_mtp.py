"""CPU regressions for slot mapping, independent MTP acceptance and worker admission."""
import queue
import threading
import types
import unittest
from dataclasses import asdict
from unittest.mock import patch

import torch

from .engine import Engine, GenParams, _BatchRequest
from .model_worker import GenerationGate, _serve_connection
from .test_worker_profile_control import Connection


def params(**kwargs):
    return GenParams(temperature=0, repetition_penalty=1, frequency_penalty=0,
                     presence_penalty=0, ban_cycles=False, loop_detect=False, **kwargs)


class BatchMTPTests(unittest.TestCase):
    def make_engine(self):
        e = Engine.__new__(Engine)
        e._request_state = threading.local()
        e.max_seqs, e.max_decode_slots, e.max_seq_len, e.mtp = 3, 2, 1048576, 3
        e._slot_lock = threading.Lock()
        e.mtp_stats = dict(steps=0, accepted=0, drafted=0)
        e._active_slots = {}
        for sid in (1, 2):
            req = _BatchRequest([1, 2], params(max_new_tokens=10), 10, None)
            req.next_token = sid
            req.out_tokens = [sid]
            req.pos = 10 * sid
            req.main_h = torch.full((3,), sid, dtype=torch.bfloat16)
            req.written_max = req.pos - 1
            e._active_slots[sid] = req
        def draft(tokens, positions, hidden, written):
            self.draft_inputs = (tokens.clone(), positions.clone(), hidden.clone(), written.clone())
            return torch.tensor([[3, 4, 5, 6, 7], [8, 9, 10, 11, 12]])
        def write(hidden, seq, positions):
            self.write_seq = seq.tolist()
            self.write_pos = positions.tolist()
        e.ds = types.SimpleNamespace(device='cpu', dim=1, block=5, targets=[0],
                                     draft_rows=draft, write_main_rows=write)
        def step(tokens, positions, seq, pmax):
            self.verifier_inputs = (tokens, positions, seq, pmax)
            result = torch.full((8, 32), -100.)
            # Slot 1 accepts one draft; slot 2 accepts all three.
            for row, token in enumerate([3, 7, 0, 0, 8, 9, 10, 11]):
                result[row, token] = 100
            return result
        e.rt = types.SimpleNamespace(B=8, step=step,
                                     main_hid={0: torch.arange(24).reshape(8, 3).to(torch.bfloat16)})
        return e

    def test_different_acceptance_lengths_keep_slot_state_separate(self):
        e = self.make_engine()
        result = e._batch_mtp_step([1, 2])
        self.assertEqual(result, [[3, 7], [8, 9, 10, 11]])
        self.assertEqual(self.verifier_inputs[2], [1] * 4 + [2] * 4)
        self.assertEqual(self.write_pos, [10, 11, 12, 13, 20, 21, 22, 23])
        self.assertEqual(e._active_slots[1].main_h.tolist(), [3, 4, 5])
        self.assertEqual(e._active_slots[2].main_h.tolist(), [21, 22, 23])
        self.assertEqual(e._active_slots[1].written_max, 13)
        self.assertEqual(e.mtp_stats['accepted'], 4)
        self.assertEqual(self.draft_inputs[1].tolist(), [9, 19])

    def test_one_active_slot_uses_smaller_verifier_and_draft_graph(self):
        e = self.make_engine()
        def draft(tokens, positions, hidden, written, seq_ids=None):
            self.assertEqual(tokens.shape[0], 1)
            self.assertEqual(seq_ids.tolist(), [2])
            return torch.tensor([[8, 9, 10, 11, 12]])
        e.ds.draft_rows = draft
        def step(tokens, positions, seq, pmax):
            self.assertEqual(seq, [2] * 4)
            self.assertEqual(tokens, [2, 8, 9, 10])
            logits = torch.full((4, 32), -100.)
            for row, token in enumerate([8, 9, 10, 11]):
                logits[row, token] = 100
            return logits
        e.rt_mtp_single = types.SimpleNamespace(B=4, step=step,
            main_hid={0: torch.arange(12).reshape(4, 3).to(torch.bfloat16)})
        result = e._batch_mtp_step([2])
        self.assertEqual(result, [[8, 9, 10, 11]])
        self.assertEqual(e.last_batch_verify_rows, 4)
        self.assertEqual(self.write_seq, [2] * 4)

    def test_draft_graph_dispatch_preserves_dynamic_slot_ids(self):
        from .dspark import DSparkRows
        ds = DSparkRows.__new__(DSparkRows)
        ds.device = 'cpu'
        ds.graphs_by_size = {}
        for size in (1, 2):
            inputs = dict(tokens=torch.zeros(size, dtype=torch.long), pos=torch.zeros(size, dtype=torch.long),
                          mh=torch.zeros(size, 3), wmax=torch.zeros(size, dtype=torch.long),
                          seq=torch.zeros(size, dtype=torch.long))
            ds.graphs_by_size[size] = (types.SimpleNamespace(replay=lambda: None), inputs, torch.zeros(size, 5), 1)
        def run(size, seq=None):
            return ds.draft_rows(torch.ones(size, dtype=torch.long), torch.zeros(size, dtype=torch.long),
                                 torch.zeros(size, 3), torch.zeros(size, dtype=torch.long), seq_ids=seq)
        self.assertEqual(run(1, torch.tensor([2])).shape, (1, 5))
        self.assertEqual(ds.graphs_by_size[1][1]['seq'].tolist(), [2])
        self.assertEqual(run(2).shape, (2, 5))
        self.assertEqual(ds.graphs_by_size[2][1]['seq'].tolist(), [1, 2])
        run(1, torch.tensor([1]))
        self.assertEqual(ds.graphs_by_size[1][1]['seq'].tolist(), [1])

    def test_context_tail_preserves_speculative_high_water(self):
        e = self.make_engine()
        e._active_slots[1].written_max = e._active_slots[1].pos + 3
        req = e._active_slots[2]
        req.pos = e.max_seq_len - 4
        req.written_max = e.max_seq_len - 1
        seen = []
        def step(tokens, positions, seq, pmax):
            seen.append((seq[0], positions[0], pmax[0]))
            return torch.arange(32).reshape(1, 32).float()
        e.rt_b1 = types.SimpleNamespace(step=step, main_hid={0:torch.zeros(1,3)})
        result = e._batch_context_tail_step([1,2])
        self.assertEqual(result, [[31],[31]])
        self.assertEqual(seen, [(1,10,13), (2,e.max_seq_len-4,e.max_seq_len-1)])
        self.assertEqual(e._active_slots[1].written_max,13)
        self.assertEqual(req.written_max,e.max_seq_len-1)

    def test_sparse_slot_and_context_tail_do_not_draft_out_of_bounds(self):
        e = self.make_engine()
        req = e._active_slots[2]
        req.pos = e.max_seq_len - 2
        req.max_new = 2
        result = e._batch_mtp_step([2])
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(self.verifier_inputs[2], [2] + [0] * 7)
        self.assertEqual(self.verifier_inputs[1], [1048574] + list(range(7)))
        self.assertEqual(self.write_seq, [2])
        self.assertEqual(self.draft_inputs[1].tolist(), [0, 0])

    def test_scheduler_stops_inside_accepted_blocks_and_releases_slots(self):
        from .streaming import IncrementalTokenDecoder
        e = self.make_engine()
        e.lock = threading.Lock()
        e._slot_tokens_lock = threading.Lock()
        e._slot_tokens = {1: [1], 2: [2]}
        e.slot_states = {1: {}, 2: {}}
        e._free_decode_slots = []
        e._batch_queue = queue.Queue()
        e._batch_stop_event = threading.Event()
        e.stats_tracker = None
        e.eos = 0
        e.tok = types.SimpleNamespace(decode=lambda ids, **kw: ''.join(map(str, ids)))
        requests = list(e._active_slots.values())
        requests[0].max_new = 3
        for req in requests:
            req.text_decoder = IncrementalTokenDecoder(e.tok)
            req.stream_queue = queue.Queue()
        def step(ids):
            e._batch_stop_event.set()
            return [[3, 4, 5], [8, 0, 9]]
        e._batch_mtp_step = step
        e._batch_worker_loop()
        self.assertEqual(requests[0].result_text, '134')
        self.assertEqual(requests[0].finish_reason, 'length')
        self.assertEqual(requests[1].result_text, '28')
        self.assertEqual(requests[1].finish_reason, 'stop')
        self.assertTrue(all(req.done_event.is_set() for req in requests))
        self.assertFalse(e._active_slots)
        self.assertEqual(e._free_decode_slots, [1, 2])
        self.assertEqual(e.mtp_stats['status'], 'completed')

    def test_prefill_seeds_only_selected_draft_slot(self):
        e = self.make_engine()
        rings = torch.arange(18).reshape(3, 3, 2).clone()
        other = rings[2].clone()
        e.ds.win = 2
        e.ds.blocks = [types.SimpleNamespace(attn=types.SimpleNamespace(window_kv_cache=rings))]
        e.model = types.SimpleNamespace(main_hidden=torch.arange(12).reshape(1, 4, 3))
        req = e._active_slots[1]
        req.prompt_ids = [1, 2, 3, 4]
        e._seed_batch_mtp(req, 1)
        self.assertTrue(torch.equal(rings[1], rings[0]))
        self.assertTrue(torch.equal(rings[2], other))
        self.assertEqual(self.write_seq, [1, 1])
        self.assertEqual(self.write_pos, [2, 3])
        self.assertEqual(req.main_h.tolist(), [9, 10, 11])
        self.assertEqual(req.written_max, 3)

    def test_request_metadata_is_thread_local(self):
        e = Engine.__new__(Engine)
        e._request_state = threading.local()
        barrier = threading.Barrier(2)
        results = {}
        def run(i):
            e.last_finish_reason = str(i)
            e._disable_prefix_cache_once = bool(i)
            barrier.wait(timeout=3)
            results[i] = (e.last_finish_reason, e._disable_prefix_cache_once)
        threads = [threading.Thread(target=run, args=(i,)) for i in (0, 1)]
        for t in threads: t.start()
        for t in threads: t.join(4)
        self.assertEqual(results, {0: ('0', False), 1: ('1', True)})

    def test_worker_admits_two_batch_requests_and_refuses_maintenance(self):
        barrier = threading.Barrier(2)
        gate = GenerationGate()
        e = Engine.__new__(Engine)
        e._request_state = threading.local()
        e.max_seqs = 3
        maintenance = []
        def generate(ids, p, **kwargs):
            barrier.wait(timeout=3)
            check = Connection(dict(protocol=1, operation='set_phase_profile', enabled=True))
            _serve_connection(check, e, gate)
            maintenance.append(check.sent[0]['kind'])
            e.last_finish_reason = 'length' if ids[0] == 1 else 'stop'
            e.last_decode_tok_s = ids[0]
            return str(ids[0]), ids[0]
        e.generate_text = generate
        conns = [Connection(dict(protocol=1, operation='generate_text', prompt_ids=[i],
                                 params=asdict(params(max_new_tokens=10)))) for i in (1, 2)]
        threads = [threading.Thread(target=_serve_connection, args=(c, e, gate)) for c in conns]
        for t in threads: t.start()
        for t in threads: t.join(4)
        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertEqual(maintenance, ['error', 'error'])
        self.assertEqual([c.sent[0]['value'] for c in conns], [('1', 1), ('2', 2)])
        self.assertEqual([c.sent[0]['finish_reason'] for c in conns], ['length', 'stop'])
        self.assertTrue(gate.acquire(blocking=False))
        gate.release()

    def test_stream_close_keeps_request_registered_until_finished(self):
        e = Engine.__new__(Engine)
        e._request_state = threading.local()
        e.max_seqs, e.max_seq_len = 3, 1048576
        e._batch_queue = queue.Queue()
        done = threading.Event()
        def consume():
            stream = e.stream_text([1], params(max_new_tokens=3))
            next(stream)
            stream.close()
            done.set()
        t = threading.Thread(target=consume)
        t.start()
        req = e._batch_queue.get(timeout=2)
        req.stream_queue.put('hello')
        self.assertFalse(done.wait(.05))
        self.assertTrue(req.cancel_event.is_set())
        req.done_event.set()
        t.join(2)
        self.assertTrue(done.is_set())


if __name__ == '__main__':
    unittest.main()
