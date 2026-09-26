"""Incremental separation of DeepSeek thinking, visible text, and DSML calls."""


class IncrementalTokenDecoder:
    """Decode complete Unicode spans while retaining partial byte-level tokens.

    Byte-level tokenizers can split one UTF-8 character across multiple token
    IDs. Decoding each ID separately turns the unfinished byte sequences into
    U+FFFD, even though decoding the whole sequence yields the right character.
    """

    def __init__(self, tokenizer, max_pending: int = 16):
        self.tokenizer = tokenizer
        self.pending: list[int] = []
        self.max_pending = max_pending

    def push(self, token_id: int) -> str:
        self.pending.append(int(token_id))
        text = self.tokenizer.decode(self.pending, errors="replace")
        if "\ufffd" in text and len(self.pending) < self.max_pending:
            return ""
        self.pending.clear()
        return text

    def flush(self) -> str:
        text = self.tokenizer.decode(self.pending, errors="replace") if self.pending else ""
        self.pending.clear()
        return text


def _safe_prefix(value: str, markers: tuple[str, ...]) -> str:
    """Hold a suffix that could become a control marker on the next token."""
    keep = 0
    for marker in markers:
        for size in range(1, min(len(marker), len(value)) + 1):
            if value.endswith(marker[:size]):
                keep = max(keep, size)
    return value[:-keep] if keep else value


class ChatStreamSplitter:
    def __init__(self, thinking: bool = False):
        self.thinking = thinking
        self.raw = ""
        self.reasoning = ""
        self.content = ""

    def push(self, piece: str) -> list[dict[str, str]]:
        self.raw += piece
        raw = self.raw
        deltas = []
        if self.thinking or "<think>" in raw:
            # The prompt often already ends with <think>, so completion text
            # begins directly with reasoning and contains no opening marker.
            if raw.startswith("<think>"):
                thought = raw[len("<think>"):]
            elif "<think>".startswith(raw):
                return []
            else:
                thought = raw
            if "</think>" in thought:
                thought, visible = thought.split("</think>", 1)
            else:
                thought = _safe_prefix(thought, ("</think>",))
                visible = None
            if thought.startswith(self.reasoning):
                delta = thought[len(self.reasoning):]
                if delta:
                    deltas.append({"reasoning_content": delta})
                    self.reasoning = thought
            if visible is None:
                return deltas
        elif "</think>" in raw:
            visible = raw.split("</think>", 1)[1]
        else:
            visible = _safe_prefix(raw, ("<think>", "</think>"))

        visible = visible.split("<｜DSML｜ calls>", 1)[0]
        visible = _safe_prefix(visible, ("<｜DSML｜ calls>",))
        if visible.startswith(self.content):
            delta = visible[len(self.content):]
            if delta:
                deltas.append({"content": delta})
                self.content = visible
        return deltas
