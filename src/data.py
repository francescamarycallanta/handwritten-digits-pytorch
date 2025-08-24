"""
Loads the MNIST data and makes DataLoaders.
Also splits some data for validation (to check progress).
"""


from typing import Tuple
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch




MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)




def build_transforms(augment: bool = False):
"""Turn images into tensors and normalize them.
Augment=True adds tiny rotations to mix things up.
"""
tfms = []
if augment:
tfms.append(transforms.RandomRotation(10)) # small shake
tfms += [
transforms.ToTensor(),
transforms.Normalize(MNIST_MEAN, MNIST_STD),
]
return transforms.Compose(tfms)




def get_loaders(
data_root: str = "data",
batch_size: int = 128,
val_split: float = 0.1,
seed: int = 42,
augment: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
"""Create train/val/test loaders.
Loaders feed the model mini-batches of data.
"""
tfm = build_transforms(augment)


trainval = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
test = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)


val_size = int(len(trainval) * val_split)
train_size = len(trainval) - val_size


gen = torch.Generator().manual_seed(seed) # makes split repeatable
train_ds, val_ds = random_split(trainval, [train_size, val_size], generator=gen)


# num_workers can be >0 on your machine; 2 is safe for most places.
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


return train_loader, val_loader, test_loader
