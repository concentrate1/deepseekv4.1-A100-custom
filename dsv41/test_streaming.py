"""Pure stream framing checks; does not load the model."""
import unittest

from dsv41.streaming import ChatStreamSplitter


class StreamSplitterTests(unittest.TestCase):
    def collect(self, thinking, pieces):
        splitter = ChatStreamSplitter(thinking)
        deltas = [delta for piece in pieces for delta in splitter.push(piece)]
        return ("".join(d.get("reasoning_content", "") for d in deltas),
                "".join(d.get("content", "") for d in deltas))

    def test_prompt_already_opened_think(self):
        reasoning, content = self.collect(True, ["先推理", "</thi", "nk>最终答案"])
        self.assertEqual((reasoning, content), ("先推理", "最终答案"))

    def test_completion_contains_opening_think(self):
        reasoning, content = self.collect(False, ["<thi", "nk>推理</th", "ink>正文"])
        self.assertEqual((reasoning, content), ("推理", "正文"))

    def test_plain_content_and_tool_marker(self):
        reasoning, content = self.collect(False, ["答", "案<｜DSML｜ ca", "lls>ignored"])
        self.assertEqual((reasoning, content), ("", "答案"))

    def test_partial_closing_marker_never_leaks(self):
        reasoning, content = self.collect(True, ["思考<", "/think", ">正文"])
        self.assertEqual((reasoning, content), ("思考", "正文"))


if __name__ == "__main__":
    unittest.main()
