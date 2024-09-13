import unittest
import torch
import torch.nn.functional as F
from code.functions.softmax import triton_softmax, naive_softmax

class TestExample(unittest.TestCase):

    def test_naive_softmax(self):
        """Test the naive softmax function."""
        sample = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.float32, device='cuda')
        ref_out = F.softmax(sample, dim=-1)
        naive_out = naive_softmax(sample)
        assert torch.allclose(ref_out, naive_out, rtol=1e-05, atol=1e-08), "Tensors are not nearly equal"

    def test_triton_softmax(self):
        """Test the triton softmax function."""
        sample = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.float32, device='cuda')
        ref_out = F.softmax(sample, dim=-1)
        triton_out = triton_softmax(sample)
        assert torch.allclose(ref_out, triton_out, rtol=1e-05, atol=1e-08), "Tensors are not nearly equal"

if __name__ == '__main__':
    unittest.main()
