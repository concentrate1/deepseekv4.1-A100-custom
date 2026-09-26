"""CPU regression checks for image spans at pipeline chunk boundaries."""
import unittest

import torch

from dsv41.vision import ImageInput, VisionTower, scatter_image_spans


class ImageChunkTests(unittest.TestCase):
    def test_complete_span_layout(self):
        tower = object.__new__(VisionTower)
        torch.nn.Module.__init__(tower)
        tower.image_start = torch.nn.Parameter(torch.tensor([1., 1.]))
        tower.image_newline = torch.nn.Parameter(torch.tensor([2., 2.]))
        tower.image_end = torch.nn.Parameter(torch.tensor([3., 3.]))
        tower.encode_image = lambda *_: torch.tensor([[4., 4.], [5., 5.]])
        image = ImageInput(start=7, patches=torch.empty(0), n_vit_h=1, n_vit_w=1,
                           types=torch.tensor([0, 1, 1, 2, 3]))
        span = tower.image_span_embeddings(image, torch.device("cpu"), torch.float32)
        self.assertEqual(span.tolist(), [[1., 1.], [4., 4.], [5., 5.], [2., 2.], [3., 3.]])

    def test_image_after_first_chunk_and_crossing_boundary(self):
        prompt = torch.zeros((1, 12, 2))
        spans = [(0, 5, torch.arange(10, dtype=torch.float32).reshape(5, 2))]
        for start in (0, 4, 8):
            chunk = prompt[:, start:start + 4]
            scatter_image_spans(chunk, spans, start)
        expected = torch.zeros_like(prompt)
        expected[:, 5:10] = spans[0][2]
        torch.testing.assert_close(prompt, expected)

    def test_multiple_images_and_samples(self):
        prompt = torch.zeros((2, 8, 1))
        spans = [(0, 1, torch.tensor([[1.], [2.]])),
                 (1, 4, torch.tensor([[3.], [4.], [5.]]))]
        for start in (0, 4):
            scatter_image_spans(prompt[:, start:start + 4], spans, start)
        self.assertEqual(prompt[0, :, 0].tolist(), [0, 1, 2, 0, 0, 0, 0, 0])
        self.assertEqual(prompt[1, :, 0].tolist(), [0, 0, 0, 0, 3, 4, 5, 0])


if __name__ == "__main__":
    unittest.main()
