"""
Tiny test to make sure the model runs.
Catches simple mistakes early.
"""

import torch
from src.models import MNISTMLP




def test_forward_shape():
m = MNISTMLP()
x = torch.randn(4, 1, 28, 28)
y = m(x)
assert y.shape == (4, 10)
