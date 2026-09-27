"""Serialized SSE writes and stoppable heartbeats for generation endpoints."""
import math
import os
import threading


class SSEWriter:
    def __init__(self, output, cancel=None, interval=None):
        if interval is None:
            # 默认每 5 秒保活，降低流式等待时的空闲断连风险；0 关闭，不改变回答内容。
            try:
                interval = float(os.environ.get("DSV41_HTTP_HEARTBEAT_S", "5"))
            except ValueError:
                interval = 5.0
        if not math.isfinite(interval) or interval < 0:
            interval = 5.0
        self.output = output
        self.cancel = cancel
        self.interval = interval
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.thread = None

    def start(self):
        if self.interval > 0:
            self.thread = threading.Thread(target=self._heartbeat, daemon=True)
            self.thread.start()
        return self

    def _write(self, data):
        try:
            self.output.write(data)
            self.output.flush()
        except OSError:
            self.stopped.set()
            if self.cancel is not None:
                self.cancel.set()
            raise

    def write(self, data):
        with self.lock:
            self._write(data)

    def finish(self, data=b"data: [DONE]\n\n"):
        with self.lock:
            self.stopped.set()
            self._write(data)

    def _heartbeat(self):
        while not self.stopped.wait(self.interval):
            try:
                with self.lock:
                    if self.stopped.is_set():
                        return
                    self._write(b": keep-alive\n\n")
            except OSError:
                return

    def close(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
