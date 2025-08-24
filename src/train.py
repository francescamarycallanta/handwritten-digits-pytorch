"""
Training code: one epoch loop, validation, and early stopping.
Early stopping means "stop if not improving".
"""
from typing import Dict, Tuple
import time
import torch
import torch.nn as nn
from .utils import accuracy_from_logits

def train_one_epoch(model, loader, optimizer, criterion, device) -> Tuple[float, float]:
    """Train for one pass over the data.
    Returns average loss and accuracy.
    """
    model.train()
  total_loss, total_correct, total = 0.0, 0, 0


  for images, labels in loader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad() # clear old gradients
      logits = model(images) # forward pass (make predictions)
      loss = criterion(logits, labels)
      loss.backward() # compute gradients
      optimizer.step() # move weights a tiny bit
    
      
      total_loss += loss.item() * images.size(0)
      total_correct += (logits.argmax(1) == labels).sum().item()
      total += labels.size(0)
    
  return total_loss / total, total_correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    """Check performance without changing the model.
    We use this for validation and testing.
    """
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0


    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)


return total_loss / total, total_correct / total

def train_model(model, train_loader, val_loader, optimizer, criterion, device,
    epochs: int = 30, patience: int = 5, save_path: str = "best.pt") -> Dict[str, list]:
    """Full training loop with early stopping.
    Saves the best model to a file.
    """
    best_val = float("inf")
    wait = 0
    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
      
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"Epoch {ep:02d} | {dt:.1f}s | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")
        
        
        if va_loss < best_val - 1e-4: # tiny margin to count as "better"
            best_val = va_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= patience:
            print(f"Early stopping at epoch {ep}. Best val loss: {best_val:.4f}")
            break

    # load best after training
    model.load_state_dict(torch.load(save_path, map_location=device))
    print("Loaded best weights from:", save_path)
    return history
