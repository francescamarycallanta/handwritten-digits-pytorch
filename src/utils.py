"""
This file holds tiny helper functions.
They make the other files shorter and cleaner.
"""


import random, os
import numpy as np
import torch




def set_seed(seed: int = 42):
"""Make runs repeatable.
Same seed -> same shuffles -> similar results.
"""
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
torch.cuda.manual_seed_all(seed)




def get_device() -> str:
"""Pick GPU if available, else CPU.
GPU makes training much faster.
"""
return "cuda" if torch.cuda.is_available() else "cpu"




def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
"""Turn model scores (logits) into predictions,
then compare to true labels to get accuracy.
"""
preds = logits.argmax(dim=1)
correct = (preds == labels).sum().item()
total = labels.size(0)
return correct / total
