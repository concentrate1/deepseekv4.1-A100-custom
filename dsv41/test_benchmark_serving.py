"""Benchmark parsing and failure accounting without a model or external network."""
import io
import json
import unittest
from unittest.mock import patch

from . import benchmark_serving as bench


class BenchmarkTests(unittest.TestCase):
    def test_sse_comments_multiline_and_truncation(self):
        self.assertEqual(list(bench.events(io.BytesIO(b': heartbeat\n\ndata: {"a":\ndata: 1}\n\ndata: [DONE]\n\n'))),
                         ['{"a":\n1}', '[DONE]'])
        with self.assertRaises(ValueError):
            list(bench.events(io.BytesIO(b'data: unfinished\n')))

    def test_stream_metrics_use_usage_not_chunk_count(self):
        chunks = [
            {'choices': [{'delta': {'role': 'assistant'}, 'finish_reason': None}]},
            {'choices': [{'delta': {'content': 'hello world'}, 'finish_reason': None}]},
            {'choices': [{'delta': {}, 'finish_reason': 'length'}]},
            {'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}},
        ]
        data = b': heartbeat\n\n' + b''.join(('data: ' + json.dumps(c) + '\n\n').encode() for c in chunks)
        data += b'data: [DONE]\n\n'
        with patch.object(bench, 'open_url', return_value=io.BytesIO(data)) as request:
            result = bench.run_request('http://local', 'model', {'name': 'a', 'messages': []}, False)
        self.assertEqual(result['usage']['completion_tokens'], 5)
        self.assertEqual(result['output']['content'], 'hello world')
        self.assertEqual(result['content_chunk_gaps_s'], [])
        self.assertIsNone(result['generation_tokens_s_estimate'])
        self.assertEqual(request.call_args.args[2]['X-DSV41-Prefix-Cache'], 'off')

    def test_long_output_and_generation_rate(self):
        case = bench.long_output_cases(128)[0]
        self.assertEqual(case['name'], 'long_output')
        self.assertIn('1 through 1000', case['messages'][-1]['content'])
        chunks = [
            {'choices': [{'delta': {'content': '1, '}, 'finish_reason': None}]},
            {'choices': [{'delta': {'content': '2'}, 'finish_reason': 'length'}],
             'usage': {'completion_tokens': 5}},
        ]
        data = b''.join(('data: ' + json.dumps(c) + '\n\n').encode() for c in chunks)
        data += b'data: [DONE]\n\n'
        with patch.object(bench, 'open_url', return_value=io.BytesIO(data)), \
             patch.object(bench.time, 'perf_counter', side_effect=[10, 12, 13, 14]):
            result = bench.run_request('http://local', 'model', case, True)
        self.assertEqual(result['first_content_s'], 2)
        self.assertEqual(result['generation_tokens_s_estimate'], 2)

    def test_error_or_missing_usage_fails_instead_of_reporting_zero_tokens(self):
        for data in (b'data: {"error":"failed"}\n\n', b'data: [DONE]\n\n'):
            with patch.object(bench, 'open_url', return_value=io.BytesIO(data)):
                with self.assertRaises((RuntimeError, ValueError)):
                    bench.run_request('http://local', 'model', {'name': 'a', 'messages': []}, True)

    def test_suite_checks_cold_concurrent_and_records_failures(self):
        def get(url, **kw):
            return {'data': [{'id': 'model'}]} if url.endswith('/v1/models') else {}
        def run(url, model, case, cache, *args):
            if case['name'] == 'broken':
                raise RuntimeError('own test')
            return dict(case=case['name'], output_sha256='same', first_content_s=.1,
                        content_chunk_gaps_s=[.01, .02], usage={'completion_tokens': 3})
        with patch.object(bench, 'get_json', side_effect=get), patch.object(bench, 'run_request', side_effect=run) as request:
            report = bench.run_suite('http://local', [{'name': 'good'}, {'name': 'broken'}], concurrency=2)
        self.assertFalse(report['consistent'])
        self.assertEqual(len(report['errors']), 7)
        self.assertEqual(len(report['phases']['concurrent_cold']['results']), 4)
        self.assertEqual(sum(not call.args[3] for call in request.call_args_list), 6)
        self.assertEqual(report['phases']['cached']['chunk_gap_p99_s'], .02)


if __name__ == '__main__':
    unittest.main()
