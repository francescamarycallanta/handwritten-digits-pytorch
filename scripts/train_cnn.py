"""
Train the small CNN from the command line.
Run: python scripts/train_cnn.py --epochs 30
"""


import argparse, os
import torch
import torch.nn as nn
from src.utils import set_seed, get_device
from src.data import get_loaders
from src.models import MNISTCNN
from src.train import train_model
from src.evaluate import test_and_confusion, plot_history, show_misclassified

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--model-path', type=str, default='mnist_cnn.pt')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    
    
    train_loader, val_loader, test_loader = get_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        augment=args.augment,
    )
    model = MNISTCNN(dropout_p=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    history = train_model(
        model, train_loader, val_loader, optimizer, criterion, device,
        epochs=args.epochs, patience=args.patience, save_path=args.model_path
    )
  
  plot_history(history)
  test_acc, conf = test_and_confusion(model, test_loader, device)
  print(f"Test accuracy: {test_acc:.4f}")
  print("Confusion matrix:\n", conf)
  show_misclassified(model, test_loader, device)

if __name__ == '__main__':
main()
