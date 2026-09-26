"""Lightweight API-side codec and IPC client; never loads model weights."""
from dataclasses import asdict
from functools import partial
from multiprocessing.reduction import ForkingPickler
import json
import os
import sys
import threading

import torch
from transformers import AutoTokenizer
from .engine import Engine
from .model_worker import call, request
from .vision import VisionConfig, parse_tagged_text, prepare_vl_inputs


def _reduce_cpu_tensor(tensor):
    """Send image inputs inline, without PyTorch's cross-process FD handshake."""
    if tensor.device.type != "cpu":
        raise ValueError("gateway IPC accepts CPU tensors only")
    return partial(torch.tensor, dtype=tensor.dtype), (tensor.tolist(),)


# multiprocessing.connection uses ForkingPickler. The standard Torch reducer
# shares an FD whose process auth key differs across independently launched
# gateway and worker processes. Inline values keep the worker independent.
ForkingPickler.register(torch.Tensor, _reduce_cpu_tensor)


class GatewayEngine:
    _fallback_parse_dsml = staticmethod(Engine._fallback_parse_dsml)
    parse_completion = Engine.parse_completion

    def __init__(self, ckpt, socket_path):
        self.socket_path = socket_path
        info = request(socket_path, "info")
        self.model_name = info["model_name"]
        self.max_seq_len = info["max_seq_len"]
        if os.path.realpath(info["ckpt"]) != os.path.realpath(ckpt):
            raise RuntimeError("worker checkpoint does not match gateway checkpoint")
        if (info["devices"] != [2, 0, 1, 3] or info["mtp"] not in (0, 3, 4, 5)
                or not 1 <= info["max_seq_len"] <= 1048576):
            raise RuntimeError("worker does not match a supported local profile")
        self.worker_pid = info["pid"]
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        sys.path.insert(0, os.path.join(ckpt, "encoding"))
        from encoding import encode_messages, parse_message_from_completion_text
        self._encode = encode_messages
        self._parse = parse_message_from_completion_text
        self.thinking_mode = "chat"
        with open(os.path.join(ckpt, "inference", "config.json")) as cfg_file:
            self.vision_config = VisionConfig.from_cfg(json.load(cfg_file))
        self._state = threading.local()

    @property
    def _disable_prefix_cache_once(self):
        return getattr(self._state, "disable_prefix_cache", False)

    @_disable_prefix_cache_once.setter
    def _disable_prefix_cache_once(self, value):
        self._state.disable_prefix_cache = bool(value)

    @property
    def last_finish_reason(self):
        return getattr(self._state, "finish_reason", "stop")

    @property
    def last_decode_tok_s(self):
        return getattr(self._state, "decode_tok_s", None)

    def format_chat(self, messages, thinking_mode=None, reasoning_effort=None):
        normalized = []
        for message in messages:
            item = dict(message)
            content = item.get("content")
            if isinstance(content, str) and "<image>" in content and "</image>" in content:
                item["content"] = parse_tagged_text(content)
            normalized.append(item)
        prompt, media = self._encode(normalized, thinking_mode=thinking_mode or self.thinking_mode,
                                     reasoning_effort=reasoning_effort, return_multi_modal_data=True)
        images_raw = media.get("images", []) if isinstance(media, dict) else []
        if not images_raw:
            return self.tok.encode(prompt), None, None
        ids, token_types, image_inputs = prepare_vl_inputs(prompt, images_raw, self.tok, self.vision_config)
        import torch
        return ids, [image_inputs], torch.tensor([token_types], dtype=torch.long)

    def _payload(self, ids, params, images, token_types):
        return {"prompt_ids": ids, "params": asdict(params), "images": images,
                "token_types": token_types,
                "disable_prefix_cache": self._disable_prefix_cache_once}

    def _remember(self, item, token_count=None):
        self._state.finish_reason = item.get("finish_reason", "stop")
        speed = item.get("decode_tok_s")
        # The already-running worker may still report the prefill-sampled
        # first token as thousands of decode tokens per second.
        if token_count is not None and token_count <= 1:
            speed = 0.0
        elif speed is not None and speed > 1000:
            speed = None
        self._state.decode_tok_s = speed

    def generate_text(self, ids, params, images=None, token_types=None, cancel_event=None):
        item = request(self.socket_path, "generate_text", cancel_event=cancel_event, **self._payload(ids, params, images, token_types))
        self._remember(item, item["value"][1])
        return item["value"]

    def _stream(self, operation, ids, params, images, token_types, cancel_event=None):
        iterator = call(self.socket_path, {"operation": operation, **self._payload(ids, params, images, token_types)}, stream=True, cancel_event=cancel_event)
        token_count = 0
        try:
            for item in iterator:
                if item["kind"] == "done":
                    self._remember(item, token_count)
                else:
                    value = item["value"]
                    if operation == "stream_text" and value[1] is not None:
                        token_count = value[1]
                    elif operation == "generate":
                        token_count += 1
                    yield value
        finally:
            iterator.close()

    def generate(self, ids, params, images=None, token_types=None):
        yield from self._stream("generate", ids, params, images, token_types)

    def stream_text(self, ids, params, images=None, token_types=None, cancel_event=None):
        yield from self._stream("stream_text", ids, params, images, token_types, cancel_event=cancel_event)

    def jev_inference(self, prompt, schema, max_batch=32):
        return request(self.socket_path, "jev", prompt=prompt, schema=schema, max_batch=max_batch)["value"]

    def get_cache_stats(self):
        value = request(self.socket_path, "cache_stats")["value"]
        # The first token is sampled from prefill logits, before speculative
        # decode begins. Excluding it avoids a huge first-frame tok/s spike.
        if value.get("mtp", {}).get("status") == "generating":
            total = 0.0
            for slot in value.get("slots", []):
                if slot.get("status") != "generating":
                    continue
                produced = slot.get("generated_tokens", 0)
                elapsed = slot.get("elapsed_s", 0.0)
                slot["tok_s"] = round((produced - 1) / elapsed, 1) if produced > 1 and elapsed >= 0.05 else 0.0
                total += slot["tok_s"]
            value["combined_decode_tok_s"] = round(total, 1)
        return value

    def worker_health(self):
        info = request(self.socket_path, "info")
        if info["pid"] != self.worker_pid or info["protocol"] != 1:
            raise RuntimeError("model worker changed; restart the gateway to reload the codec")
        return {"status": "ok", "worker_pid": self.worker_pid}
